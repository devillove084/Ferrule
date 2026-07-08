//! Runtime/graph execution summaries and benchmark counters.
//!
//! These structs are deliberately backend- and model-family-neutral. A graph
//! backend, legacy runner, or CLI command can fill whichever counters it knows,
//! while JSON output remains stable for local benchmarking and regression
//! tracking.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use ferrule_runtime::backend_object_store::{BackendObject, BackendObjectStore};
use ferrule_runtime::graph_program::GraphProgram;
use ferrule_runtime::graph_runtime::{ExternalBindingKind, ExternalResidency};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphProgramSummary {
    pub name: String,
    pub layers: usize,
    pub nodes: usize,
    pub values: usize,
    pub inputs: usize,
    pub outputs: usize,
    pub externals: usize,
    pub artifact_tensor_bindings: usize,
    pub artifact_group_bindings: usize,
    pub expert_registry_bindings: usize,
    pub kv_state_bindings: usize,
    pub device_resident_bindings: usize,
    pub streamable_bindings: usize,
    pub backend_managed_bindings: usize,
    pub ops: BTreeMap<String, usize>,
    pub artifact_groups: BTreeMap<String, usize>,
}

impl GraphProgramSummary {
    #[allow(dead_code)]
    pub fn from_program(program: &GraphProgram) -> Self {
        let mut summary = Self {
            name: program.graph.name().unwrap_or("<unnamed>").to_string(),
            layers: program.runtime_plan.layer_count(),
            nodes: program.graph.nodes().len(),
            values: program.graph.values().len(),
            inputs: program.graph.inputs().len(),
            outputs: program.graph.outputs().len(),
            externals: program.bindings.len(),
            ..Self::default()
        };

        for node in program.graph.nodes() {
            *summary
                .ops
                .entry(format!("{}::{}", node.op().domain(), node.op().name()))
                .or_default() += 1;
        }

        for binding in program.bindings.entries() {
            match binding.kind {
                ExternalBindingKind::Weight | ExternalBindingKind::ArtifactTensor => {
                    summary.artifact_tensor_bindings += 1;
                }
                ExternalBindingKind::ArtifactGroup(group) => {
                    summary.artifact_group_bindings += 1;
                    *summary
                        .artifact_groups
                        .entry(group.as_str().to_string())
                        .or_default() += 1;
                }
                ExternalBindingKind::ExpertRegistry => summary.expert_registry_bindings += 1,
                ExternalBindingKind::KvState => summary.kv_state_bindings += 1,
                ExternalBindingKind::ResidentExpert
                | ExternalBindingKind::Adapter
                | ExternalBindingKind::Speculation
                | ExternalBindingKind::Other(_) => {}
            }
            match binding.residency {
                ExternalResidency::Device => summary.device_resident_bindings += 1,
                ExternalResidency::Streamable => summary.streamable_bindings += 1,
                ExternalResidency::BackendManaged => summary.backend_managed_bindings += 1,
                ExternalResidency::Host | ExternalResidency::Paged => {}
            }
        }

        summary
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendObjectSummary {
    pub objects: usize,
    pub artifact_tensors: usize,
    pub artifact_groups: usize,
    pub artifact_group_tensors: usize,
    pub expert_registries: usize,
    pub registered_experts: usize,
    pub kv_states: usize,
    pub opaque_objects: usize,
    pub artifact_bytes: u64,
    pub expert_load_source_bytes: u64,
}

impl BackendObjectSummary {
    #[allow(dead_code)]
    pub fn from_store(store: &BackendObjectStore) -> Self {
        let mut summary = Self {
            objects: store.len(),
            ..Self::default()
        };
        for object in store.objects().values() {
            match object {
                BackendObject::ArtifactTensor(slice) => {
                    summary.artifact_tensors += 1;
                    summary.artifact_bytes = summary.artifact_bytes.saturating_add(slice.bytes);
                }
                BackendObject::ArtifactGroup(group) => {
                    summary.artifact_groups += 1;
                    summary.artifact_group_tensors += group.tensors.len();
                    summary.artifact_bytes = summary
                        .artifact_bytes
                        .saturating_add(group.tensors.iter().map(|tensor| tensor.bytes).sum());
                }
                BackendObject::ExpertRegistry(registry) => {
                    summary.expert_registries += 1;
                    summary.registered_experts += registry.experts.len();
                    summary.expert_load_source_bytes =
                        summary.expert_load_source_bytes.saturating_add(
                            registry
                                .experts
                                .values()
                                .map(|source| source.bytes())
                                .sum::<u64>(),
                        );
                }
                BackendObject::KvState(_) => summary.kv_states += 1,
                BackendObject::Opaque { .. } => summary.opaque_objects += 1,
            }
        }
        summary
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTimingCounters {
    pub load_us: u64,
    pub artifact_materialize_us: u64,
    pub layer_bind_us: u64,
    pub prefill_us: u64,
    pub decode_us: u64,
    pub graph_execute_us: u64,
    /// MoE calls observed in the CUDA operator path.
    pub moe_calls: u64,
    pub moe_tc_calls: u64,
    pub moe_scalar_calls: u64,
    pub moe_reduce_calls: u64,
    /// Env-gated GPU-synchronized MoE timings. These stay zero unless
    /// `FERRULE_CUDA_MOE_TIMING=1` is set for the run.
    pub moe_total_us: u64,
    pub moe_pointer_upload_us: u64,
    pub moe_input_prepare_us: u64,
    pub moe_gate_up_us: u64,
    pub moe_swiglu_us: u64,
    pub moe_hidden_pack_us: u64,
    pub moe_down_us: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelCounters {
    pub launches: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferCounters {
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub host_to_device_copies: u64,
    pub device_to_host_copies: u64,
    pub artifact_uploads: u64,
    pub artifact_upload_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertRuntimeCounters {
    pub loads: u64,
    pub load_bytes: u64,
    pub evictions: u64,
    pub selected: u64,
    pub resident_experts: usize,
    pub resident_expert_bytes: u64,
    pub host_cache_hits: u64,
    pub host_cache_misses: u64,
    pub host_cache_evictions: u64,
    pub host_cache_entries: usize,
    pub host_cache_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryCounters {
    pub peak_host_bytes: Option<u64>,
    pub peak_device_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeCounters {
    pub timing: RuntimeTimingCounters,
    pub kernels: KernelCounters,
    pub transfers: TransferCounters,
    pub experts: ExpertRuntimeCounters,
    pub memory: MemoryCounters,
}

#[allow(dead_code)]
impl RuntimeCounters {
    pub fn record_load(&mut self, elapsed: Duration) {
        self.timing.load_us = self.timing.load_us.saturating_add(duration_us(elapsed));
    }

    pub fn record_artifact_materialize(&mut self, elapsed: Duration, bytes: u64) {
        self.timing.artifact_materialize_us = self
            .timing
            .artifact_materialize_us
            .saturating_add(duration_us(elapsed));
        self.transfers.host_to_device_bytes =
            self.transfers.host_to_device_bytes.saturating_add(bytes);
        self.transfers.artifact_uploads = self.transfers.artifact_uploads.saturating_add(1);
        self.transfers.artifact_upload_bytes =
            self.transfers.artifact_upload_bytes.saturating_add(bytes);
    }

    pub fn record_layer_bind(&mut self, elapsed: Duration) {
        self.timing.layer_bind_us = self
            .timing
            .layer_bind_us
            .saturating_add(duration_us(elapsed));
    }

    pub fn record_prefill(&mut self, elapsed: Duration) {
        self.timing.prefill_us = self.timing.prefill_us.saturating_add(duration_us(elapsed));
    }

    pub fn record_decode(&mut self, elapsed: Duration) {
        self.timing.decode_us = self.timing.decode_us.saturating_add(duration_us(elapsed));
    }

    pub fn record_graph_execute(&mut self, elapsed: Duration) {
        self.timing.graph_execute_us = self
            .timing
            .graph_execute_us
            .saturating_add(duration_us(elapsed));
    }

    pub fn record_kernel_launches(&mut self, launches: u64) {
        self.kernels.launches = self.kernels.launches.saturating_add(launches);
    }

    pub fn record_host_to_device(&mut self, bytes: u64) {
        self.transfers.host_to_device_copies =
            self.transfers.host_to_device_copies.saturating_add(1);
        self.transfers.host_to_device_bytes =
            self.transfers.host_to_device_bytes.saturating_add(bytes);
    }

    pub fn record_device_to_host(&mut self, bytes: u64) {
        self.transfers.device_to_host_copies =
            self.transfers.device_to_host_copies.saturating_add(1);
        self.transfers.device_to_host_bytes =
            self.transfers.device_to_host_bytes.saturating_add(bytes);
    }

    pub fn record_artifact_uploads(&mut self, uploads: u64, bytes: u64) {
        self.transfers.artifact_uploads = self.transfers.artifact_uploads.saturating_add(uploads);
        self.transfers.artifact_upload_bytes =
            self.transfers.artifact_upload_bytes.saturating_add(bytes);
    }

    pub fn record_expert_loads(&mut self, loads: u64, bytes: u64) {
        self.experts.loads = self.experts.loads.saturating_add(loads);
        self.experts.load_bytes = self.experts.load_bytes.saturating_add(bytes);
    }

    pub fn record_expert_evictions(&mut self, evictions: u64) {
        self.experts.evictions = self.experts.evictions.saturating_add(evictions);
    }

    pub fn set_expert_host_cache(
        &mut self,
        hits: u64,
        misses: u64,
        evictions: u64,
        entries: usize,
        bytes: u64,
    ) {
        self.experts.host_cache_hits = hits;
        self.experts.host_cache_misses = misses;
        self.experts.host_cache_evictions = evictions;
        self.experts.host_cache_entries = entries;
        self.experts.host_cache_bytes = bytes;
    }

    pub fn record_selected_experts(&mut self, selected: u64) {
        self.experts.selected = self.experts.selected.saturating_add(selected);
    }

    pub fn set_expert_residency(&mut self, resident_experts: usize, resident_bytes: u64) {
        self.experts.resident_experts = resident_experts;
        self.experts.resident_expert_bytes = resident_bytes;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeBenchSummary {
    pub graph: Option<GraphProgramSummary>,
    pub objects: Option<BackendObjectSummary>,
    pub counters: RuntimeCounters,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub kernel_launches_per_token: f64,
    pub host_to_device_bytes_per_token: f64,
    pub device_to_host_bytes_per_token: f64,
    pub expert_loads_per_token: f64,
}

impl RuntimeBenchSummary {
    pub fn new(
        graph: Option<GraphProgramSummary>,
        objects: Option<BackendObjectSummary>,
        counters: RuntimeCounters,
        prompt_tokens: usize,
        generated_tokens: usize,
    ) -> Self {
        let total_tokens = prompt_tokens.saturating_add(generated_tokens).max(1) as f64;
        let prefill_tok_per_s = rate(prompt_tokens, counters.timing.prefill_us);
        let decode_tok_per_s = rate(generated_tokens, counters.timing.decode_us);
        Self {
            graph,
            objects,
            kernel_launches_per_token: counters.kernels.launches as f64 / total_tokens,
            host_to_device_bytes_per_token: counters.transfers.host_to_device_bytes as f64
                / total_tokens,
            device_to_host_bytes_per_token: counters.transfers.device_to_host_bytes as f64
                / total_tokens,
            expert_loads_per_token: counters.experts.loads as f64 / total_tokens,
            counters,
            prompt_tokens,
            generated_tokens,
            prefill_tok_per_s,
            decode_tok_per_s,
        }
    }
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn rate(tokens: usize, elapsed_us: u64) -> f64 {
    if tokens == 0 || elapsed_us == 0 {
        0.0
    } else {
        tokens as f64 / (elapsed_us as f64 / 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_counters_compute_per_token_rates() {
        let mut counters = RuntimeCounters::default();
        counters.timing.prefill_us = 2_000_000;
        counters.timing.decode_us = 500_000;
        counters.kernels.launches = 30;
        counters.record_host_to_device(120);
        counters.record_device_to_host(60);
        counters.record_expert_loads(3, 1024);

        let summary = RuntimeBenchSummary::new(None, None, counters, 4, 6);
        assert_eq!(summary.prefill_tok_per_s, 2.0);
        assert_eq!(summary.decode_tok_per_s, 12.0);
        assert_eq!(summary.kernel_launches_per_token, 3.0);
        assert_eq!(summary.host_to_device_bytes_per_token, 12.0);
        assert_eq!(summary.device_to_host_bytes_per_token, 6.0);
        assert_eq!(summary.expert_loads_per_token, 0.3);
    }
}
