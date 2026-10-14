//! Checked dense BF16 device budget; no CUDA context or payload reads in planning.
use super::{PipelineConfig, PipelineStageDescription, error};
use crate::parallel::pipeline::PhysicalKvLayout;
use ferrule_common::Result;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    Attention, Bf16DenseCudaProfile, BoundDecoderResources, FeedForward, LayerSegmentPlan,
};

/// Conservative peak for the synchronous single-stage native-BF16 program.
/// Device weights have no F32 resident mirror. Host checkpoint payloads remain
/// owned by the existing materializer; this is a device, not host-RSS budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseBf16MemoryPlan {
    pub weight_bytes: u64,
    /// Upper bound: cos/sin BF16 tables per layer (identical tables may be shared).
    pub rope_bytes: u64,
    pub kv_bytes: u64,
    pub bf16_rows_bytes: u64,
    /// GEMM output, expanded attention Q/output, and LM-head host-export staging.
    pub f32_work_bytes: u64,
    pub metadata_bytes: u64,
    /// Current BF16 GEMM provider requires zero extra workspace; checked on owner.
    pub gemm_workspace_bytes: u64,
    /// Allocator granularity, context growth and transient-retirement headroom.
    pub headroom_bytes: u64,
    pub total_device_bytes: u64,
}

fn sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |a, &b| a.checked_add(b))
        .filter(|&n| n <= isize::MAX as u64)
        .ok_or_else(|| error("BF16 device budget overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as u64)
        .ok_or_else(|| error("BF16 device budget overflow"))
}

impl DenseBf16MemoryPlan {
    pub fn new(
        resources: &BoundDecoderResources,
        plan: &LayerSegmentPlan,
        config: PipelineConfig,
    ) -> Result<Self> {
        config.validate()?;
        Bf16DenseCudaProfile::validate(resources)?;
        let spec = resources.spec();
        if config.precision != ExecutionPrecisionPolicy::bf16_compatibility()
            || plan.total_layers() != spec.layers().len()
            || plan.layers() != (0..spec.layers().len())
            || !plan.owns_embedding()
            || !plan.owns_output()
            || spec
                .max_sequence_length()
                .is_some_and(|n| config.max_positions > n)
        {
            return Err(error(
                "BF16 runtime admits only a complete single standard stage within checkpoint context",
            ));
        }
        resources
            .validate_parameter_limits(config.max_parameter_bytes, config.max_parameter_bytes)?;
        let mut seen = std::collections::BTreeSet::new();
        let mut weight_bytes = 0;
        for p in resources.state_dict().parameters() {
            if seen.insert(p.canonical_id()) {
                weight_bytes = sum(&[weight_bytes, p.weight().slice().bytes])?;
            }
        }
        let h = spec.hidden_size() as u64;
        let v = spec.vocab_size() as u64;
        let r = config.max_batch_tokens as u64;
        let mut bf16_rows_bytes = mul(mul(r, sum(&[mul(8, h)?, v])?)?, 2)?;
        let mut f32_work_bytes = mul(mul(r, h.max(v))?, 4)?;
        let mut rope_bytes = 0;
        let mut geometry = None;
        for layer in spec.layers() {
            let Attention::Gqa(g) = layer.attention() else {
                unreachable!("profile validated")
            };
            let FeedForward::SwiGlu(f) = layer.feed_forward() else {
                unreachable!("profile validated")
            };
            let current = (g.num_kv_heads(), g.head_dim());
            if geometry.is_some_and(|previous| previous != current) {
                return Err(error("BF16 pipeline requires uniform typed KV geometry"));
            }
            geometry = Some(current);
            let q = g.query().out_features() as u64;
            let k = g.key().out_features() as u64;
            let i = f.intermediate_size() as u64;
            // Sum over live families rather than assuming fused scratch reuse:
            // residual/normalizations, Q/K/V before and after RoPE, gate/up/product,
            // attention/projected output and BF16 logits. Layers execute serially.
            let widths = sum(&[mul(8, h)?, mul(8, q)?, mul(8, k)?, mul(4, i)?, v])?;
            bf16_rows_bytes = bf16_rows_bytes.max(mul(mul(r, widths)?, 2)?);
            f32_work_bytes = f32_work_bytes.max(mul(
                mul(r, sum(&[mul(2, q)?, h.max(v).max(q).max(k).max(i)])?)?,
                4,
            )?);
            rope_bytes = sum(&[
                rope_bytes,
                mul(
                    mul(
                        config.max_positions as u64,
                        g.rotary().region().dimensions() as u64,
                    )?,
                    2,
                )?,
            ])?;
            for width in [h, v, q, k, i] {
                if mul(r, width)? > i32::MAX as u64 {
                    return Err(error("BF16 row buffer exceeds native i32 extent"));
                }
            }
        }
        let (kv_heads, head_dim) = geometry.ok_or_else(|| error("empty BF16 stage"))?;
        let physical_pages = config.physical_pages(PhysicalKvLayout::standard(
            ferrule_model::ModelExecutionBackend::Cuda,
        ))?;
        let description = PipelineStageDescription {
            plan: plan.clone(),
            config,
            physical_pages,
            hidden: spec.hidden_size(),
            vocabulary: spec.vocab_size(),
            kv_heads,
            head_dim,
            expert_group: None,
        };
        let kv_bytes = description.physical_bytes()?;
        // Guard uploads slots/offsets, three row arrays and status; positions and
        // token ids are also transient device arrays. The transport uses bounded
        // 64-KiB pinned slots; F32 export device staging is charged above.
        let metadata_bytes = sum(&[mul(physical_pages as u64, 16)?, mul(r, 32)?, 4])?;
        let subtotal = sum(&[
            weight_bytes,
            rope_bytes,
            kv_bytes,
            bf16_rows_bytes,
            f32_work_bytes,
            metadata_bytes,
        ])?;
        let headroom_bytes = sum(&[64 << 20, subtotal / 10])?;
        Ok(Self {
            weight_bytes,
            rope_bytes,
            kv_bytes,
            bf16_rows_bytes,
            f32_work_bytes,
            metadata_bytes,
            gemm_workspace_bytes: 0,
            headroom_bytes,
            total_device_bytes: sum(&[subtotal, headroom_bytes])?,
        })
    }

    pub fn check_available(self, free_bytes: usize) -> Result<()> {
        if self.total_device_bytes > free_bytes as u64 {
            return Err(error(format!(
                "dense BF16 requires {} device bytes including headroom, only {free_bytes} free",
                self.total_device_bytes
            )));
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(super) fn validate_provider(resources: &BoundDecoderResources, rows: usize) -> Result<()> {
        use ferrule_backend::cuda::operators::linear::{
            Bf16GemmLayout, bf16_gemm_workspace_requirements,
        };
        let check = |l: &ferrule_model::transformer::Linear| -> Result<()> {
            for m in [1, rows] {
                let workspace = bf16_gemm_workspace_requirements(Bf16GemmLayout::contiguous(
                    m,
                    l.out_features(),
                    l.in_features(),
                ))?;
                if workspace.bytes != 0 {
                    return Err(error(
                        "BF16 provider workspace changed; revise explicit device budget",
                    ));
                }
            }
            Ok(())
        };
        check(resources.spec().output())?;
        for layer in resources.spec().layers() {
            let Attention::Gqa(g) = layer.attention() else {
                unreachable!("profile validated")
            };
            let FeedForward::SwiGlu(f) = layer.feed_forward() else {
                unreachable!("profile validated")
            };
            for l in [
                g.query(),
                g.key(),
                g.value(),
                g.output(),
                f.gate(),
                f.up(),
                f.down(),
            ] {
                check(l)?;
            }
        }
        Ok(())
    }
}
