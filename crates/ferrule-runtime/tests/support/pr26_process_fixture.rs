//! Fixed CPU checkpoint from process_decoder's deterministic synthetic fixture.
//! Kept separate so PR26 does not edit PR16's wire/projection tests. This is not
//! a production-model benchmark or an alternative transport implementation.
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{DecoderRecipe, LayerSegmentPlan, SyntheticDecoderRecipe};
use ferrule_runtime::parallel::pipeline::{PipelineConfig, PipelineParallelExecutor};
use ferrule_runtime::parallel::process::decoder::*;
use ferrule_runtime::parallel::process::*;
use serde_json::json;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[path = "build_process_child.rs"]
mod build_process_child;

pub fn launch_mode(mode: &str) -> ProcessLaunch {
    static CHILD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    let fixture = CHILD.get_or_init(|| {
        std::env::var_os("FERRULE_PR26_CHILD")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                build_process_child::build("ferrule-runtime", "example", "process_rank_child")
            })
    });
    assert!(fixture.is_file());
    ProcessLaunch::new(fixture).arg(mode).arg("30000")
}
fn launch() -> ProcessLaunch {
    launch_mode("decoder")
}
pub fn options() -> ProcessOwnerConfig {
    ProcessOwnerConfig {
        startup_timeout: Duration::from_secs(10),
        command_timeout: Duration::from_secs(3),
        terminate_grace: Duration::from_millis(20),
        kill_grace: Duration::from_secs(1),
        ..Default::default()
    }
}
pub fn identity(rank: u32) -> ProcessIdentity {
    ProcessIdentity::new(
        ProcessGroupEpoch::new(1).unwrap(),
        ParallelRankId::new(rank),
        ProcessOwnerInstanceId::new(u64::from(rank) + 1).unwrap(),
    )
}
pub fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}
pub fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 16,
        max_positions: 16,
        max_batch_tokens: 16,
        session_capacity: 3,
        max_parameter_bytes: 4096,
        max_ack_polls: 8,
        precision: ExecutionPrecisionPolicy::f32(),
    }
}
pub fn topology(degree: u32) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(71),
        degree,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, 1, 1, 1, 1, degree as usize).unwrap(),
    )
    .unwrap()
}

pub struct Fixture(PathBuf);
impl Fixture {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-pr26-decoder-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        let config = json!({
            "vocab_size":8, "hidden_size":4, "num_attention_heads":2,
            "num_key_value_heads":1, "head_dim":2, "intermediate_size":4,
            "num_experts":2, "experts_per_token":2, "max_position_embeddings":16,
            "rms_norm_eps":0.00001, "rope_theta":10000.0, "tie_word_embeddings":false
        });
        std::fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let recipe = SyntheticDecoderRecipe::new().build(&config).unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for parameter in recipe
            .schema()
            .parameters()
            .iter()
            .filter(|p| p.alias_of().is_none())
        {
            let seed = parameter
                .path()
                .as_str()
                .bytes()
                .fold(0usize, |sum, byte| (sum * 31 + byte as usize) % 251);
            let start = payload.len();
            for index in 0..parameter.shape().iter().product::<usize>() {
                let value = if parameter.shape().len() == 1 {
                    0.9 + index as f32 * 0.03
                } else {
                    ((seed + index * 17 + index * index * 3) % 41) as f32 * 0.012 - 0.24
                };
                payload.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(SyntheticDecoderRecipe::external_name(parameter.path()), json!({"dtype":"F32", "shape":parameter.shape(), "data_offsets":[start,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(path.join("model.safetensors"), file).unwrap();
        Self(path)
    }
    pub fn boots(&self, degree: u32, ep: bool) -> Vec<(ProcessIdentity, DecoderBoot)> {
        (0..degree)
            .map(|rank| {
                let layers = if degree == 1 {
                    0..2
                } else {
                    rank as usize..rank as usize + 1
                };
                let experts = ep.then(|| {
                    let first = 10 + rank * 2;
                    ExpertPlacementFrame {
                        source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
                        source: rank,
                        members: vec![first, first + 1],
                        entries: layers
                            .clone()
                            .flat_map(|layer| [(layer, 0, first), (layer, 1, first + 1)])
                            .collect(),
                        max_tokens: 32,
                        max_bytes: 4096,
                        devices: vec![DecoderDevice::Cpu; 2],
                        timeout_ms: 3000,
                    }
                });
                (
                    identity(rank),
                    DecoderBoot {
                        version: DECODER_WIRE_VERSION,
                        rank,
                        checkpoint: self.0.clone(),
                        recipe: DecoderRecipeKind::Synthetic,
                        segment: SegmentFrame::encode(
                            &LayerSegmentPlan::new(2, layers, rank == 0, rank + 1 == degree)
                                .unwrap(),
                        ),
                        precision: DecoderPrecision::F32,
                        device: DecoderDevice::Cpu,
                        kv: KvConfigFrame::encode(config()),
                        experts,
                    },
                )
            })
            .collect()
    }
    pub fn pipeline(
        &self,
        degree: u32,
        ep: bool,
    ) -> (PipelineParallelExecutor, SharedProcessPipelineTransport) {
        let boots = self.boots(degree, ep);
        let plans = boots
            .iter()
            .map(|(_, boot)| boot.segment.decode().unwrap())
            .collect();
        let transport = ProcessPipelineTransport::spawn(launch(), boots, options())
            .unwrap()
            .shared();
        let pipeline = PipelineParallelExecutor::new_with_external_expert_transport(
            topology(degree),
            plans,
            config(),
            transport.clone(),
        )
        .unwrap();
        (pipeline, transport)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
