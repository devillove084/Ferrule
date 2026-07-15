//! DeepSeek-V4 expert-I/O prediction and immutable residency snapshots.

use std::sync::Arc;

use ferrule_common::expert_io::{ExpertIoEstimate, ExpertIoPhase};
use ferrule_common::{Error, Result};

use crate::moe::prediction::{ExpertAccessPhase, ExpertPredictContext, ExpertResidency};
use crate::runner::ExpertIoModelRunner;

use super::runner::DeepSeekV4Runner;
use super::sequence::DeepSeekV4SequenceExecutionState;

#[derive(Debug, Clone)]
pub struct DeepSeekV4ExpertIoLayerSnapshot {
    source_bytes: Arc<[u64]>,
    source_order: Arc<[usize]>,
    residency: Box<[ExpertResidency]>,
}

impl DeepSeekV4ExpertIoLayerSnapshot {
    pub fn new(source_bytes: Arc<[u64]>, residency: Box<[ExpertResidency]>) -> Self {
        let mut source_order = (0..source_bytes.len()).collect::<Vec<_>>();
        source_order.sort_unstable_by_key(|&expert| std::cmp::Reverse(source_bytes[expert]));
        Self::with_source_order(source_bytes, source_order.into(), residency)
    }

    pub(crate) fn with_source_order(
        source_bytes: Arc<[u64]>,
        source_order: Arc<[usize]>,
        residency: Box<[ExpertResidency]>,
    ) -> Self {
        Self {
            source_bytes,
            source_order,
            residency,
        }
    }

    pub fn expert_count(&self) -> usize {
        self.source_bytes.len().min(self.residency.len())
    }

    pub fn source_bytes(&self, expert: usize) -> Option<u64> {
        self.source_bytes.get(expert).copied()
    }

    pub fn source_order(&self) -> &[usize] {
        &self.source_order
    }

    pub fn residency(&self) -> &[ExpertResidency] {
        &self.residency
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekV4ExpertIoSnapshot {
    layers: Box<[DeepSeekV4ExpertIoLayerSnapshot]>,
    prediction_budget: usize,
}

impl DeepSeekV4ExpertIoSnapshot {
    pub fn new(layers: Box<[DeepSeekV4ExpertIoLayerSnapshot]>, prediction_budget: usize) -> Self {
        Self {
            layers,
            prediction_budget,
        }
    }

    pub fn layers(&self) -> &[DeepSeekV4ExpertIoLayerSnapshot] {
        &self.layers
    }

    pub const fn prediction_budget(&self) -> usize {
        self.prediction_budget
    }
}

#[derive(Debug)]
pub struct DeepSeekV4ExpertIoAdmission {
    dense_experts: Vec<usize>,
}

#[derive(Debug)]
pub struct DeepSeekV4ExpertIoBatchState {
    snapshot: DeepSeekV4ExpertIoSnapshot,
    layer_offsets: Vec<usize>,
    admitted: Vec<bool>,
    candidate_epochs: Vec<u32>,
    next_epoch: u32,
}

impl DeepSeekV4ExpertIoBatchState {
    pub fn new(snapshot: DeepSeekV4ExpertIoSnapshot) -> Self {
        let mut layer_offsets = Vec::with_capacity(snapshot.layers().len());
        let mut total = 0usize;
        for layer in snapshot.layers() {
            layer_offsets.push(total);
            total = total.saturating_add(layer.expert_count());
        }
        Self {
            snapshot,
            layer_offsets,
            admitted: vec![false; total],
            candidate_epochs: vec![0; total],
            next_epoch: 0,
        }
    }

    fn begin_candidate(&mut self) -> u32 {
        let epoch = self.next_epoch.wrapping_add(1);
        if epoch == 0 {
            self.candidate_epochs.fill(0);
            self.next_epoch = 1;
        } else {
            self.next_epoch = epoch;
        }
        self.next_epoch
    }

    fn estimate(
        &mut self,
        state: &DeepSeekV4SequenceExecutionState,
        phase: ExpertIoPhase,
        token_ids: &[u32],
    ) -> Result<(ExpertIoEstimate, DeepSeekV4ExpertIoAdmission)> {
        let access_phase = match phase {
            ExpertIoPhase::Prefill => ExpertAccessPhase::Prefill,
            ExpertIoPhase::Decode => ExpertAccessPhase::Decode,
        };
        let epoch = self.begin_candidate();
        let mut estimate = ExpertIoEstimate {
            confidence: 0.0,
            ..ExpertIoEstimate::default()
        };
        let mut confidence = 1.0f32;
        let mut predicted = 0usize;
        let mut has_unknown_demand = false;
        let mut dense_experts = Vec::new();

        for (layer_index, layer) in self.snapshot.layers().iter().enumerate() {
            let forecast_budget = self
                .snapshot
                .prediction_budget()
                .saturating_mul(token_ids.len().max(1))
                .min(layer.expert_count());
            let forecast = state.expert_predictor().forecast(
                ExpertPredictContext {
                    layer: layer_index,
                    phase: access_phase,
                    budget: forecast_budget,
                    num_experts: layer.expert_count(),
                    residency: layer.residency(),
                },
                true,
            );
            let unknown = forecast_budget.saturating_sub(forecast.len());
            for prediction in forecast {
                let dense = self.layer_offsets[layer_index]
                    .checked_add(prediction.expert.expert)
                    .ok_or_else(|| Error::Internal("expert-I/O dense index overflow".into()))?;
                if dense >= self.admitted.len()
                    || self.admitted[dense]
                    || self.candidate_epochs[dense] == epoch
                {
                    continue;
                }
                self.candidate_epochs[dense] = epoch;
                dense_experts.push(dense);
                predicted = predicted.saturating_add(1);
                confidence = confidence.min(prediction.confidence);
                let bytes = layer
                    .source_bytes(prediction.expert.expert)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "missing source bytes for expert {}:{}",
                            layer_index, prediction.expert.expert
                        ))
                    })?;
                account_residency(
                    &mut estimate,
                    layer
                        .residency()
                        .get(prediction.expert.expert)
                        .copied()
                        .unwrap_or(ExpertResidency::Cold),
                    bytes,
                );
            }
            if unknown > 0 {
                account_unknown_demand(&mut estimate, layer, unknown);
                has_unknown_demand = true;
            }
        }
        if predicted > 0 && !has_unknown_demand {
            estimate.confidence = confidence;
        }
        Ok((estimate, DeepSeekV4ExpertIoAdmission { dense_experts }))
    }

    fn admit(&mut self, admission: DeepSeekV4ExpertIoAdmission) {
        for dense in admission.dense_experts {
            if let Some(admitted) = self.admitted.get_mut(dense) {
                *admitted = true;
            }
        }
    }
}

impl ExpertIoModelRunner for DeepSeekV4Runner {
    type ExpertIoBatchState = DeepSeekV4ExpertIoBatchState;
    type ExpertIoAdmission = DeepSeekV4ExpertIoAdmission;

    fn begin_expert_io_batch(&self) -> Self::ExpertIoBatchState {
        DeepSeekV4ExpertIoBatchState::new(self.expert_io_snapshot())
    }

    fn estimate_expert_io(
        &self,
        batch: &mut Self::ExpertIoBatchState,
        sequence: &Self::SequenceState,
        phase: ExpertIoPhase,
        token_ids: &[u32],
    ) -> Result<(ExpertIoEstimate, Self::ExpertIoAdmission)> {
        batch.estimate(sequence, phase, token_ids)
    }

    fn admit_expert_io(
        &self,
        batch: &mut Self::ExpertIoBatchState,
        admission: Self::ExpertIoAdmission,
    ) {
        batch.admit(admission);
    }
}

fn account_unknown_demand(
    estimate: &mut ExpertIoEstimate,
    layer: &DeepSeekV4ExpertIoLayerSnapshot,
    mut experts: usize,
) {
    for target in [
        ExpertResidency::Cold,
        ExpertResidency::HostStaged,
        ExpertResidency::Materializing,
        ExpertResidency::GpuReady,
    ] {
        for &expert in layer.source_order() {
            if experts == 0 {
                return;
            }
            if layer.residency().get(expert).copied() == Some(target) {
                account_residency(estimate, target, layer.source_bytes(expert).unwrap_or(0));
                experts -= 1;
            }
        }
    }
}

fn account_residency(estimate: &mut ExpertIoEstimate, residency: ExpertResidency, bytes: u64) {
    match residency {
        ExpertResidency::GpuReady => {}
        ExpertResidency::Materializing => {
            estimate.inflight_reusable_bytes =
                estimate.inflight_reusable_bytes.saturating_add(bytes);
        }
        ExpertResidency::HostStaged => {
            estimate.incremental_unique_bytes =
                estimate.incremental_unique_bytes.saturating_add(bytes);
        }
        ExpertResidency::Cold => {
            estimate.incremental_unique_bytes =
                estimate.incremental_unique_bytes.saturating_add(bytes);
            estimate.predicted_cold_bytes = estimate.predicted_cold_bytes.saturating_add(bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residency_accounting_uses_only_truthful_transfer_classes() {
        let mut estimate = ExpertIoEstimate::default();
        account_residency(&mut estimate, ExpertResidency::GpuReady, 10);
        account_residency(&mut estimate, ExpertResidency::Materializing, 20);
        account_residency(&mut estimate, ExpertResidency::HostStaged, 30);
        account_residency(&mut estimate, ExpertResidency::Cold, 40);

        assert_eq!(estimate.incremental_unique_bytes, 70);
        assert_eq!(estimate.predicted_cold_bytes, 40);
        assert_eq!(estimate.inflight_reusable_bytes, 20);
        assert_eq!(estimate.resident_union_bytes, 0);
        assert_eq!(estimate.inflight_reads, 0);
        assert_eq!(estimate.upload_slots, 0);
    }

    #[test]
    fn cold_start_prefill_scales_conservative_cost_by_rows() {
        let layer = DeepSeekV4ExpertIoLayerSnapshot::new(
            Arc::from([10, 40, 20, 30]),
            vec![ExpertResidency::Cold; 4].into_boxed_slice(),
        );
        let snapshot = DeepSeekV4ExpertIoSnapshot::new(vec![layer].into_boxed_slice(), 1);
        let state = DeepSeekV4SequenceExecutionState::new(Vec::new(), 4);
        let mut batch = DeepSeekV4ExpertIoBatchState::new(snapshot);

        let (estimate, admission) = batch
            .estimate(&state, ExpertIoPhase::Prefill, &[11, 12])
            .unwrap();
        assert_eq!(estimate.incremental_unique_bytes, 70);
        assert_eq!(estimate.predicted_cold_bytes, 70);
        assert_eq!(estimate.confidence, 0.0);

        batch.admit(admission);
        let (repeated, _) = batch
            .estimate(&state, ExpertIoPhase::Prefill, &[11, 12])
            .unwrap();
        assert_eq!(repeated.incremental_unique_bytes, 70);
    }
}
