//! Prepared CUDA stage; the ordinary segment owns the only layer composition.

use std::rc::Rc;

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelRankId, Result};

use crate::decoder::{CudaKvView, PackedDecoderBatch};
use crate::execution::ExecutionPrecisionPolicy;
use crate::transformer::expert_parallel::RoutedSwiGluExecutor;
use crate::transformer::{
    Attention, BoundDecoderResources, BoundParameter, DeviceSegmentInput, DeviceSegmentOutput,
    LayerSegmentPlan, SegmentError, SegmentInput, SegmentOutput, SegmentResult, SegmentStage,
    StandardDecoderSegment,
};

use super::{
    CudaStandardDecoderOperators, CudaStandardKvBinding, validate_precision, validate_rope,
};

#[derive(Debug)]
pub struct CudaStandardDecoderSegment {
    segment: StandardDecoderSegment,
    operators: CudaStandardDecoderOperators,
}

impl StandardDecoderSegment {
    pub fn prepare_cuda(
        resources: &BoundDecoderResources,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
        max_positions: usize,
        max_parameter_bytes: u64,
        ops: Rc<CudaOperators>,
    ) -> SegmentResult<CudaStandardDecoderSegment> {
        let prepare = |source| SegmentError::Execution {
            stage: SegmentStage::Prepare,
            source,
        };
        validate_precision(precision).map_err(prepare)?;
        // Check only the assigned descriptors, before materializing weights.
        if let Some(layers) = resources.spec().layers().get(plan.layers()) {
            for layer in layers {
                if let Attention::Gqa(gqa) = layer.attention() {
                    validate_rope(gqa.rotary()).map_err(prepare)?;
                }
            }
        }
        let segment = Self::prepare(
            resources,
            plan,
            precision,
            max_positions,
            max_parameter_bytes,
        )?;
        let mut operators = CudaStandardDecoderOperators::new(ops, precision, segment.parameters())
            .map_err(prepare)?;
        segment
            .prepare_cuda_bindings(&mut operators)
            .map_err(prepare)?;
        Ok(CudaStandardDecoderSegment { segment, operators })
    }
}

impl CudaStandardDecoderSegment {
    pub fn precision(&self) -> ExecutionPrecisionPolicy {
        ExecutionPrecisionPolicy::f32()
    }
    pub fn activation_dtype(&self) -> crate::transformer::RowsDType {
        crate::transformer::RowsDType::F32
    }
    pub fn kv_element_type(&self) -> ferrule_common::execution::KvElementType {
        ferrule_common::execution::KvElementType::F32
    }

    pub fn plan(&self) -> &LayerSegmentPlan {
        self.segment.plan()
    }
    pub fn parameters(&self) -> &[BoundParameter] {
        self.segment.parameters()
    }
    pub fn operators(&self) -> &CudaStandardDecoderOperators {
        &self.operators
    }
    pub fn needs_quarantine(&self) -> bool {
        self.operators.needs_quarantine()
    }
    pub fn quiesce(&mut self) -> Result<()> {
        self.operators.quiesce()
    }

    /// Same-owner device input/output. KV-local planes follow the segment range.
    /// Every operator fences before returning; errors still require KV rollback.
    pub fn execute_cuda_bound(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        kv: &mut CudaKvView,
    ) -> SegmentResult<DeviceSegmentOutput> {
        let mut binding =
            CudaStandardKvBinding::new(kv, batch).map_err(|source| SegmentError::Execution {
                stage: SegmentStage::Input,
                source,
            })?;
        self.segment
            .execute_bound(batch, input, &mut binding, &mut self.operators)
    }

    /// Explicit pinned PP input/output staging, not a per-layer copy loop.
    pub fn execute(
        &mut self,
        batch: &PackedDecoderBatch,
        input: SegmentInput,
        kv: &mut CudaKvView,
    ) -> SegmentResult<SegmentOutput> {
        self.execute_cuda_bound(batch, input.into(), kv)?
            .into_host(&mut self.operators)
    }

    pub fn download_output(&mut self, output: DeviceSegmentOutput) -> SegmentResult<SegmentOutput> {
        output.into_host(&mut self.operators)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_cuda_bound_with_experts(
        &mut self,
        batch: &PackedDecoderBatch,
        input: DeviceSegmentInput,
        kv: &mut CudaKvView,
        context: (ExecutionTransactionId, ParallelRankId, &[u64]),
        executor: &mut dyn RoutedSwiGluExecutor,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> SegmentResult<DeviceSegmentOutput> {
        let mut binding =
            CudaStandardKvBinding::new(kv, batch).map_err(|source| SegmentError::Execution {
                stage: SegmentStage::Input,
                source,
            })?;
        self.segment.execute_bound_with_experts(
            batch,
            input,
            &mut binding,
            &mut self.operators,
            context,
            executor,
            check_active,
        )
    }
}
