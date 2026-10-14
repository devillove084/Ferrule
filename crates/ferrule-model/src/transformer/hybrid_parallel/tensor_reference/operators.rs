//! Rank-local math and communication; preparation owns all payload conversion.
use super::*;
use crate::{TensorRole, nn::ParameterResidency};

pub(super) struct TensorOperators<'a, 'b> {
    pub cpu: CpuStandardDecoderOperators,
    pub layout: &'a HybridRankLayout,
    pub channel: &'a RefCell<Channel<'b>>,
}
impl TensorOperators<'_, '_> {
    fn collect(
        &self,
        semantic: HybridCollectiveSemantic,
        plan: TensorParallelStagePlan,
        partial: Vec<f32>,
        count: usize,
        layer: Option<usize>,
    ) -> Result<Vec<f32>> {
        let packed = plan.pack_local_output(
            ParallelRankId::new(self.layout.lane() as u32),
            &partial,
            count,
        )?;
        let values =
            self.channel
                .borrow_mut()
                .exchange(semantic, layer, plan.collective(), packed)?;
        plan.unpack_collective_output(&values, count)
    }
}
impl StandardDecoderOperators for TensorOperators<'_, '_> {
    fn backend_name(&self) -> &'static str {
        "hybrid-rank-local-cpu-reference"
    }
    fn precision(&self) -> ExecutionPrecisionPolicy {
        ExecutionPrecisionPolicy::f32()
    }
    fn attention_heads(&self, q: usize, kv: usize) -> Result<(usize, usize)> {
        Ok((
            q / self.layout.members().len(),
            (kv / self.layout.members().len()).max(1),
        ))
    }
    fn linear(
        &mut self,
        linear: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let local = ready(self.cpu.linear(linear, input, arena)?)?.into_host()?;
        let PreparedLinearStorage::CpuLocal(matrix) = linear.storage() else {
            return Err(invalid("rank image requires prepared local CPU storage"));
        };
        if let Some(plan) = &matrix.collective {
            let count = local.shape().rows();
            let values = self.collect(
                if linear.role() == &TensorRole::OutputHead {
                    HybridCollectiveSemantic::OutputHead
                } else {
                    HybridCollectiveSemantic::AttentionOutput
                },
                plan.clone().into(),
                local.into_values(),
                count,
                match linear.parameter().residency() {
                    ParameterResidency::Layer { layer } => Some(*layer),
                    _ => None,
                },
            )?;
            rows(values, count, linear.out_features(), arena)
        } else {
            Ok(OperatorProgress::Ready(Rows::Host(local)))
        }
    }
    fn dense_swiglu(
        &mut self,
        expert: &PreparedSwiGlu,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let local = ready(self.cpu.dense_swiglu(expert, input, arena)?)?.into_host()?;
        let PreparedLinearStorage::CpuLocal(matrix) = expert.down().storage() else {
            return Err(invalid("rank MLP requires prepared local CPU storage"));
        };
        let plan = matrix
            .collective
            .as_ref()
            .ok_or_else(|| invalid("missing down reduction"))?;
        let count = local.shape().rows();
        let layer = match expert.down().parameter().residency() {
            ParameterResidency::Layer { layer } => Some(*layer),
            _ => return Err(invalid("MLP must belong to a layer")),
        };
        let semantic = match expert.down().role() {
            TensorRole::SharedExpertDown => HybridCollectiveSemantic::SharedExpertDown,
            TensorRole::DenseMlpDown => HybridCollectiveSemantic::DenseMlpDown,
            _ => return Err(invalid("unexpected MLP reduction role")),
        };
        let values = self.collect(
            semantic,
            plan.clone().into(),
            local.into_values(),
            count,
            layer,
        )?;
        rows(values, count, expert.output_width(), arena)
    }
    fn gated_delta_net(
        &mut self,
        request: GatedDeltaNetRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        let HybridLocalAttention::GatedDelta { shape, .. } =
            self.layout.attention(request.layer)?
        else {
            return Err(invalid("GDN geometry missing"));
        };
        self.cpu.gated_delta_net(GatedDeltaNetRequest {
            shape: *shape,
            ..request
        })
    }
    fn routed_swiglu(
        &mut self,
        _: usize,
        _: &Rows,
        _: &RouterRoutes,
        _: &mut dyn ExpertProvider,
        _: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        Err(invalid(
            "TP routed execution requires explicit leader EP injection; no local fallback",
        ))
    }
    fn lm_head(
        &mut self,
        head: &PreparedLinear,
        input: &Rows,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.linear(head, input, arena)
    }
    fn embedding(
        &mut self,
        embedding: &PreparedEmbedding,
        tokens: &[u32],
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.cpu.embedding(embedding, tokens, arena)
    }
    fn rms_norm(
        &mut self,
        norm: &PreparedNorm,
        input: &Rows,
        heads: usize,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        self.cpu.rms_norm(norm, input, heads, arena)
    }
    fn rope(
        &mut self,
        descriptor: &RotaryEmbedding,
        table: &PreparedRope,
        input: Rows,
        heads: usize,
        positions: &[usize],
    ) -> Result<OperatorProgress<Rows>> {
        self.cpu.rope(descriptor, table, input, heads, positions)
    }
    fn paged_gqa(
        &mut self,
        kv: &mut dyn KvView,
        request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        self.cpu.paged_gqa(kv, request)
    }
    fn router(
        &mut self,
        input: &Rows,
        policy: &MoeRouterSpec,
    ) -> Result<OperatorProgress<RouterRoutes>> {
        self.cpu.router(input, policy)
    }
    fn residual(&mut self, input: Rows, update: &Rows) -> Result<OperatorProgress<Rows>> {
        self.cpu.residual(input, update)
    }
    fn unpack_gated_query(
        &mut self,
        input: Rows,
        heads: usize,
        dim: usize,
    ) -> Result<OperatorProgress<(Rows, Rows)>> {
        self.cpu.unpack_gated_query(input, heads, dim)
    }
    fn sigmoid_gate(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
        self.cpu.sigmoid_gate(input, gate)
    }
    fn shared_expert_gate(&mut self, input: Rows, gate: &Rows) -> Result<OperatorProgress<Rows>> {
        self.cpu.shared_expert_gate(input, gate)
    }
}
