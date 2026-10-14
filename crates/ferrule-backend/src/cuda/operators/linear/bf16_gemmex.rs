//! Explicit prepared BF16-output operator and bounded sequential shape views.
use super::{CudaBf16Buffer, CudaNumericFp8Artifact, CudaOperators};
use crate::cuda::{
    operator_owner::cu,
    operators::OperatorOwner,
    providers::bf16_gemmex::{NativeBf16GemmEx, invalid},
    runtime::DeviceBuffer,
};
pub use crate::cuda_plan::bf16_gemmex::Bf16GemmExPlan;
use ferrule_common::Result;
use std::sync::Arc;

/// Owns the private cuBLAS handle, exact stream and its permanently bound storage.
/// Preparation allocates/warmups outside capture. Execution never allocates or
/// rebinds workspace. Noncaptured operands follow normal last-consumer rules.
pub struct CudaBf16GemmEx {
    pub(crate) native: Arc<NativeBf16GemmEx>,
}
impl CudaBf16GemmEx {
    pub fn plan(&self) -> Bf16GemmExPlan {
        self.native.plan
    }
    pub fn provider_version(&self) -> i32 {
        self.native.version
    }
    pub fn provider_observed_growth_bytes(&self) -> u64 {
        self.native.growth
    }
    pub fn is_poisoned(&self) -> bool {
        self.native.poisoned.get() || self.native.stream.context().graph_completion_unknown()
    }
    /// Preparation can fail after submitting warmup work, before returning an
    /// owner. The caller must quarantine its image/claim on terminal Unknown,
    /// while ordinary allocation or capability rejection remains retryable.
    pub fn preparation_completion_unknown(op: &CudaOperators) -> bool {
        op.stream_clone().context().graph_completion_unknown()
    }
    pub fn execute(
        &mut self,
        op: &CudaOperators,
        weight: &CudaNumericFp8Artifact,
        input: &CudaBf16Buffer,
        output: &mut CudaBf16Buffer,
    ) -> Result<()> {
        op.submit_operator(3, |stream| {
            self.native.launch(
                stream,
                weight,
                input.as_device_buffer(),
                output.as_device_buffer(),
                false,
            )
        })
    }
    /// Captures one projection and retains every device allocation and the
    /// handle/workspace lease through graph retirement. The graph can outlive
    /// this prepared object and the original operand wrappers.
    pub fn capture(
        &mut self,
        op: &CudaOperators,
        weight: &CudaNumericFp8Artifact,
        input: &CudaBf16Buffer,
        output: &mut CudaBf16Buffer,
    ) -> Result<CudaBf16GemmExGraph> {
        let stream = op.stream_clone();
        self.native.args(
            &stream,
            weight,
            input.as_device_buffer(),
            output.as_device_buffer(),
        )?;
        cu(stream.context().check_graph_ready())?;
        let (w, s) = weight.buffers();
        let leases = (
            cu(w.slice(0, w.len()))?,
            cu(s.slice(0, s.len()))?,
            cu(input
                .as_device_buffer()
                .slice(0, input.as_device_buffer().len()))?,
            cu(output
                .as_device_buffer()
                .slice(0, output.as_device_buffer().len()))?,
        );
        let captured = op.capture_decode_graph(|| {
            op.submit_operator(3, |stream| {
                self.native.launch(
                    stream,
                    weight,
                    input.as_device_buffer(),
                    output.as_device_buffer(),
                    true,
                )
            })
        });
        let graph = match captured {
            Ok(graph) => graph,
            Err(error) => {
                if self.native.stream.context().graph_completion_unknown() {
                    self.native.poisoned.set(true);
                    std::mem::forget(leases);
                    std::mem::forget(Arc::clone(&self.native));
                }
                return Err(error);
            }
        };
        Ok(CudaBf16GemmExGraph {
            graph: Some(graph),
            native: Arc::clone(&self.native),
            leases: Some(leases),
        })
    }
}
/// Retains graph operands plus handle/workspace. Replays only on its exact owner.
/// Callers must order overwrites of surviving operand aliases after consumers.
pub struct CudaBf16GemmExGraph {
    graph: Option<crate::cuda::graph::CudaGraphHandle>,
    native: Arc<NativeBf16GemmEx>,
    leases: Option<(
        DeviceBuffer<u8>,
        DeviceBuffer<u8>,
        DeviceBuffer<u16>,
        DeviceBuffer<u16>,
    )>,
}
impl CudaBf16GemmExGraph {
    pub fn launch(&self, op: &CudaOperators) -> Result<()> {
        self.native.owner(&op.stream_clone())?;
        let result = op.launch_graph(
            self.graph
                .as_ref()
                .ok_or_else(|| invalid("retired graph"))?,
        );
        if result.is_err() && self.native.stream.context().graph_completion_unknown() {
            self.native.poisoned.set(true);
        }
        result
    }
    pub fn upload(&self, op: &CudaOperators) -> Result<()> {
        self.native.owner(&op.stream_clone())?;
        let result = op.upload_graph(
            self.graph
                .as_ref()
                .ok_or_else(|| invalid("retired graph"))?,
        );
        if result.is_err() && self.native.stream.context().graph_completion_unknown() {
            self.native.poisoned.set(true);
        }
        result
    }
    /// Copy the native BF16 result through the existing owner download boundary.
    pub fn download(&self, op: &CudaOperators) -> Result<Vec<f32>> {
        self.native.owner(&op.stream_clone())?;
        let y = &self
            .leases
            .as_ref()
            .ok_or_else(|| invalid("retired graph"))?
            .3;
        cu(y.check_context(self.native.stream.context(), "GemmEx graph download"))?;
        // Shared runtime boundary retains DMA destination/source on either
        // copy or fence failure, and distinguishes preflight from submission.
        let host = cu(y.to_host_vec(&self.native.stream))?;
        Ok(host
            .into_iter()
            .map(|word| f32::from_bits(u32::from(word) << 16))
            .collect())
    }
}
impl Drop for CudaBf16GemmExGraph {
    fn drop(&mut self) {
        if self.native.poisoned.get()
            || self.native.stream.context().is_capturing()
            || self.native.stream.synchronize().is_err()
        {
            self.native.poisoned.set(true);
            self.native.stream.context().mark_graph_unknown();
        }
        // Graph destruction itself may lose custody. Observe its status before
        // any operand or provider lease can retire.
        drop(self.graph.take());
        if self.native.stream.context().graph_completion_unknown() {
            self.native.poisoned.set(true);
            std::mem::forget(self.leases.take());
            std::mem::forget(Arc::clone(&self.native));
        }
    }
}

impl CudaOperators {
    /// One permanently bound allocation/handle for an explicit finite set of
    /// sequential shape views. All views warm up before invocation or capture.
    pub fn prepare_bf16_gemmex_views(
        &self,
        views: &[Bf16GemmExPlan],
        storage_budget: usize,
        provider_budget: usize,
    ) -> Result<CudaBf16GemmEx> {
        if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":0:0") {
            return Err(invalid("shape pool requires CUBLAS_WORKSPACE_CONFIG=:0:0"));
        }
        let bytes = views
            .iter()
            .map(|p| p.storage_bytes())
            .max()
            .ok_or_else(|| invalid("empty shape set"))?;
        if crate::cuda_allocation_grant_bytes(bytes).is_none_or(|grant| grant > storage_budget)
            || provider_budget < crate::cuda_plan::bf16_gemmex::BF16_GEMMEX_PROVIDER_BYTES
        {
            return Err(invalid("shared shape budget"));
        }
        self.prepare_bf16_gemmex_plans(views)
    }
}
impl CudaBf16GemmEx {
    /// No allocation, synchronization, handle creation or workspace rebinding.
    /// Owner and operands must survive the caller's last-consumer proof.
    pub fn execute_view(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weight: &CudaNumericFp8Artifact,
        input: &CudaBf16Buffer,
        output: &mut CudaBf16Buffer,
    ) -> Result<()> {
        op.submit_operator(3, |s| {
            self.native.launch_view(
                s,
                plan,
                weight,
                input.as_device_buffer(),
                output.as_device_buffer(),
            )
        })
    }
    /// Packed resident BF16 A/B/C using a warmed shape in this same owner pool.
    /// No decode/copy/staging or allocation. Strided views and unretained capture
    /// are rejected; caller retains operands through its last-consumer proof.
    pub fn execute_bf16_view(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weight: &CudaBf16Buffer,
        input: &CudaBf16Buffer,
        output: &mut CudaBf16Buffer,
        layout: super::Bf16GemmLayout,
    ) -> Result<()> {
        if layout
            != super::Bf16GemmLayout::contiguous(plan.rows(), plan.layout().n, plan.layout().k)
        {
            return Err(invalid("resident GemmEx requires packed plan strides"));
        }
        op.submit_operator(1, |stream| {
            self.native.launch_resident(
                stream,
                plan,
                weight.as_device_buffer(),
                input.as_device_buffer(),
                output.as_device_buffer(),
            )
        })
    }

    /// Native full-width gate/up GEMM from two original encoded artifacts.
    /// The two outputs are a bit-preserving row-wise split, not separate GEMMs.
    pub fn execute_pair(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weights: [&CudaNumericFp8Artifact; 2],
        input: &CudaBf16Buffer,
        outputs: [&mut CudaBf16Buffer; 2],
    ) -> Result<()> {
        op.submit_operator(5, |s| {
            self.native.launch_pair::<false>(
                s,
                plan,
                weights,
                input.as_device_buffer(),
                [outputs[0].as_device_buffer(), outputs[1].as_device_buffer()],
            )
        })
    }
}

impl CudaBf16GemmEx {
    /// Decode immutable numeric FP8 into an exact packed N*K BF16 destination.
    /// Uses the original BF16-RNE numeric decode, without host reads, allocation,
    /// synchronization or GEMM. Eager-only; caller retains both operands through
    /// completion and quarantines them on terminal Unknown, as for shape views.
    pub fn decode_numeric_bf16_into(
        &mut self,
        op: &CudaOperators,
        weight: &CudaNumericFp8Artifact,
        target: &mut CudaBf16Buffer,
    ) -> Result<()> {
        op.submit_operator(1, |stream| {
            self.native
                .decode_into(stream, weight, target.as_device_buffer())
        })
    }

    /// Decode equal-layout gate/up artifacts directly into packed [2*N,K].
    /// Both sources and the entire exact-size target are validated before either
    /// launch. No temporary weight, copy, allocation or synchronization; caller
    /// retains sources/target through completion and quarantines on Unknown.
    pub fn decode_numeric_bf16_pair_into(
        &mut self,
        op: &CudaOperators,
        weights: [&CudaNumericFp8Artifact; 2],
        target: &mut CudaBf16Buffer,
    ) -> Result<()> {
        op.submit_operator(2, |stream| {
            self.native
                .decode_pair_into(stream, weights, target.as_device_buffer())
        })
    }

    /// Reuse contiguous [gate rows, up rows] BF16 weights with the original
    /// prepared full-2N GemmEx and bit-preserving split. Input must be exactly
    /// packed M*K; each output is packed M*N. No decode or activation packing.
    /// Eager-only; operands obey the same last-consumer contract as shape views.
    pub fn execute_bf16_pair_view(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weight: &CudaBf16Buffer,
        input: &CudaBf16Buffer,
        outputs: [&mut CudaBf16Buffer; 2],
        layout: super::Bf16GemmLayout,
    ) -> Result<()> {
        if layout
            != super::Bf16GemmLayout::contiguous(plan.rows(), plan.layout().n, plan.layout().k)
        {
            return Err(invalid(
                "resident pair requires packed full-2N plan strides",
            ));
        }
        op.submit_operator(2, |stream| {
            self.native.launch_resident_pair::<false>(
                stream,
                plan,
                weight.as_device_buffer(),
                input.as_device_buffer(),
                [outputs[0].as_device_buffer(), outputs[1].as_device_buffer()],
            )
        })
    }
}

impl CudaBf16GemmEx {
    /// Original two-artifact decode and prepared full-2N GEMM, followed by one
    /// extract/SiLU-BF16-RNE/product-BF16-RNE kernel into caller storage.
    /// Input/output must be exact packed [M,K]/[M,N]; plan is the warmed 2N plan.
    /// No allocation, synchronization, extra handle or intermediate output views.
    /// Eager-only; retain all operands through last consumer, quarantine on Unknown.
    pub fn execute_pair_swiglu_view_into(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weights: [&CudaNumericFp8Artifact; 2],
        input: &CudaBf16Buffer,
        activation: &mut CudaBf16Buffer,
    ) -> Result<()> {
        op.submit_operator(5, |stream| {
            self.native.launch_pair::<true>(
                stream,
                plan,
                weights,
                input.as_device_buffer(),
                [activation.as_device_buffer(); 2],
            )
        })
    }

    /// Resident packed [gate rows, up rows] weights, unchanged prepared 2N GEMM
    /// and the same two-rounding SwiGLU as `swiglu_bf16`, into exact [M,N].
    /// Layout describes the full [M,2N,K] GEMM, not the activation output.
    /// Same eager owner/workspace/lifetime/Unknown contract as the cold entry.
    pub fn execute_bf16_pair_swiglu_view_into(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weight: &CudaBf16Buffer,
        input: &CudaBf16Buffer,
        activation: &mut CudaBf16Buffer,
        layout: super::Bf16GemmLayout,
    ) -> Result<()> {
        if layout
            != super::Bf16GemmLayout::contiguous(plan.rows(), plan.layout().n, plan.layout().k)
        {
            return Err(invalid(
                "resident pair requires packed full-2N plan strides",
            ));
        }
        op.submit_operator(2, |stream| {
            self.native.launch_resident_pair::<true>(
                stream,
                plan,
                weight.as_device_buffer(),
                input.as_device_buffer(),
                [activation.as_device_buffer(); 2],
            )
        })
    }
}
