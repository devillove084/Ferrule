//! Image-directory admission for numeric rank shards, separate from TP1 caches.
use std::sync::Arc;

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::Result;

use super::super::numeric_binding::rank::{RankBindings, ResidentRankLinear};
use super::{Bindings, cuda_error};
use crate::transformer::{
    CudaNumericFp8RankBinding, NumericFp8Precision, PreparedNumericFp8RankLinear,
    hybrid_parallel::HybridRankLayout,
};

#[cfg(test)]
#[path = "../../../../tests/unit/transformer/numeric_shard_binding.rs"]
mod tests;

impl Bindings {
    pub fn configure_numeric_rank(
        &mut self,
        layout: Arc<HybridRankLayout>,
        limit: usize,
    ) -> Result<()> {
        if !self.numeric_fp8
            || self.tensor.is_some()
            || self.expert_admission.is_some()
            || !self.linear.is_empty()
            || limit == 0
        {
            return Err(cuda_error(
                "numeric rank directory requires an unused numeric image and positive grant budget",
            ));
        }
        if let Some(rank) = &self.numeric_rank {
            if rank.layout != layout || rank.limit != limit {
                return Err(cuda_error(
                    "numeric rank directory cannot change layout/budget",
                ));
            }
            return Ok(());
        }
        self.numeric_rank = Some(RankBindings {
            layout,
            limit,
            grant_bytes: 0,
            linears: Vec::new(),
        });
        Ok(())
    }

    pub fn preflight_numeric_rank(
        &self,
        plan: &crate::transformer::NumericFp8RankPlan,
        max_rows: usize,
        scratch: usize,
        precision: NumericFp8Precision,
    ) -> Result<(
        Option<usize>,
        crate::transformer::NumericFp8RankExecutionBudget,
    )> {
        let binding = plan.binding();
        let allowed = self
            .numeric_rank_allowed
            .get(&(binding.id(), binding.role().clone()))
            .ok_or_else(|| cuda_error("numeric rank parameter outside image directory"))?;
        let rank = self
            .numeric_rank
            .as_ref()
            .ok_or_else(|| cuda_error("configure numeric rank before upload"))?;
        plan.validate_binding(allowed, &rank.layout)?;
        let budget = plan.cuda_execution_budget(max_rows, scratch, precision)?;
        let existing = rank.linears.iter().enumerate().find(|(_, r)| {
            r.plan.binding().id() == binding.id() && r.plan.binding().role() == binding.role()
        });
        if let Some((slot, resident)) = existing {
            resident.plan.validate_binding(binding, plan.layout())?;
            if !resident.matches_execution(max_rows, scratch, precision) {
                return Err(cuda_error(
                    "numeric rank execution capacity/precision changed",
                ));
            }
            return Ok((Some(slot), budget));
        }
        rank.grant_bytes
            .checked_add(budget.resident_grant_bytes)
            .filter(|n| *n <= rank.limit)
            .ok_or_else(|| cuda_error("numeric rank physical residency grant budget exceeded"))?;
        Ok((None, budget))
    }

    pub fn prepare_numeric_rank(
        &mut self,
        ops: &CudaOperators,
        prepared: &PreparedNumericFp8RankLinear,
        max_rows: usize,
        scratch: usize,
        precision: NumericFp8Precision,
    ) -> Result<CudaNumericFp8RankBinding> {
        let plan = prepared.plan();
        let (existing, budget) = self.preflight_numeric_rank(plan, max_rows, scratch, precision)?;
        let rank = self.numeric_rank.as_mut().expect("preflighted directory");
        let token = |slot| CudaNumericFp8RankBinding {
            generation: self.generation,
            slot,
            plan: plan.clone(),
        };
        if let Some(slot) = existing {
            return Ok(token(slot));
        }
        let total = rank
            .grant_bytes
            .checked_add(budget.resident_grant_bytes)
            .filter(|n| *n <= rank.limit)
            .ok_or_else(|| cuda_error("numeric rank physical residency grant budget exceeded"))?;
        let payload_total = self
            .bytes
            .checked_add(plan.budget().payload_bytes)
            .ok_or_else(|| cuda_error("numeric rank resident payload overflow"))?;
        let before = ops.allocator_metrics().granted_bytes;
        let resident = ResidentRankLinear::upload(ops, prepared, max_rows, scratch, precision)?;
        // The budget uses the allocator's real size classes, not logical buffer
        // lengths. Fail closed if the actual allocator contract ever diverges.
        let observed = ops.allocator_metrics().granted_bytes.checked_sub(before);
        if observed != Some(budget.resident_grant_bytes as u64) {
            return Err(cuda_error(
                "numeric rank actual physical grants differ from admission",
            ));
        }
        let slot = rank.linears.len();
        rank.linears.push(resident);
        rank.grant_bytes = total;
        self.bytes = payload_total;
        Ok(token(slot))
    }

    pub fn numeric_rank_linear(
        &self,
        token: &CudaNumericFp8RankBinding,
    ) -> Result<&ResidentRankLinear> {
        if token.generation != self.generation {
            return Err(cuda_error(
                "numeric rank token belongs to another image generation",
            ));
        }
        let rank = self
            .numeric_rank
            .as_ref()
            .ok_or_else(|| cuda_error("missing numeric rank directory"))?;
        let resident = rank
            .linears
            .get(token.slot)
            .ok_or_else(|| cuda_error("invalid numeric rank slot"))?;
        resident
            .plan
            .validate_binding(token.plan.binding(), token.plan.layout())?;
        Ok(resident)
    }

    pub fn numeric_rank_linear_mut(
        &mut self,
        token: &CudaNumericFp8RankBinding,
    ) -> Result<&mut ResidentRankLinear> {
        self.numeric_rank_linear(token)?;
        Ok(&mut self
            .numeric_rank
            .as_mut()
            .expect("validated directory")
            .linears[token.slot])
    }
}
