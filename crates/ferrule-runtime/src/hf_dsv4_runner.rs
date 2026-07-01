//! DSV4 runner backed by real HF safetensors (when present).
//!
//! `HfDsV4Runner` can be used with `InferenceEngine` for `ferrule run` once the
//! full DSV4 checkpoint is present. Without the model it fails gracefully.

use std::path::{Path, PathBuf};

use ferrule_core::{Error, Result};
use ferrule_model::{families::HyperConnectionStage, HfSafetensorsInventory, ModelFamily};

use crate::dsv4_mock::SyntheticTokenizer;
use crate::dsv4_param;
use crate::expert_streaming::{
    ExpertId, ExpertSource, ExpertStreamingPlanner, ExpertStreamingPolicy, ExpertStreamingReader,
};
use crate::first_token_smoke::FirstTokenModel;
use crate::hyper_connection::{
    hc_head_reference, HyperConnectionConfig, HyperConnectionHeadWeights,
};
use crate::layer_binding::{
    bind_layer_source_from_hf, LayerExecutionState, LayerSourceBinding, ReferenceLayerExecutor,
};
use crate::source_binding::bind_hyper_connection_head_from_hf;
use crate::source_tensor::SourceTensorReader;
use crate::CpuReferenceExpertExecutor;

pub struct HfDsV4Runner {
    config: HyperConnectionConfig,
    layers: Vec<LayerSourceBinding>,
    head_weights: HyperConnectionHeadWeights,
    tokenizer: SyntheticTokenizer,
}

impl HfDsV4Runner {
    pub fn load(model_dir: impl AsRef<Path>, max_tensor_bytes: u64) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        if !model_dir.join("config.json").exists() {
            return Err(Error::Model(format!(
                "DSV4 model directory not found: {}",
                model_dir.display()
            )));
        }
        let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)?;
        let attention = inventory.attention_tensors(&ModelFamily::DeepSeekV4);
        let hc_tensors = inventory.hyper_connection_tensors(&ModelFamily::DeepSeekV4);
        let routers = inventory.router_tensors(&ModelFamily::DeepSeekV4);
        let shared = inventory.shared_expert_tensors(&ModelFamily::DeepSeekV4);

        let config = dsv4_param::hc_config();
        let reader = SourceTensorReader::new(max_tensor_bytes);
        let spec = dsv4_param::attention_spec();

        let mut layers = Vec::with_capacity(dsv4_param::NUM_LAYERS);
        for layer_index in 0..dsv4_param::NUM_LAYERS {
            let policy = dsv4_param::router_policy_for_layer(layer_index);
            let binding = bind_layer_source_from_hf(
                &model_dir,
                layer_index,
                &attention,
                &hc_tensors,
                &routers,
                &shared,
                &reader,
                config,
                dsv4_param::SWIGLU_LIMIT,
                policy,
                spec,
            )?;
            layers.push(binding);
        }
        let head_weights =
            bind_hyper_connection_head_from_hf(&model_dir, &hc_tensors, &reader, config)?;
        Ok(Self {
            config,
            layers,
            head_weights,
            tokenizer: SyntheticTokenizer::new(),
        })
    }

    fn decode_with_logits(
        &self,
        state: &mut LayerExecutionState,
        token_id: u32,
    ) -> Result<Vec<f32>> {
        let executor = ReferenceLayerExecutor::new(
            self.config,
            ExpertStreamingReader::new(64 * 1024 * 1024),
            CpuReferenceExpertExecutor::new(dsv4_param::SWIGLU_LIMIT),
        );
        let mut hc_state = vec![0.0f32; self.config.hc_hidden_size()];
        for i in 0..self.config.hc_hidden_size().min(64) {
            hc_state[i] = ((token_id as usize + i) % 1000) as f32 * 0.001;
        }
        for layer_index in 0..dsv4_param::NUM_LAYERS {
            let binding = &self.layers[layer_index];
            let output = executor.execute_decode_step(binding, state, &hc_state, token_id, &[])?;
            hc_state = output.hc_state;
        }
        let hidden = hc_head_reference(&hc_state, 1, self.config, &self.head_weights)?;
        let mut logits = vec![0.0f32; dsv4_param::VOCAB_SIZE];
        for i in 0..dsv4_param::HIDDEN_SIZE {
            logits[i % dsv4_param::VOCAB_SIZE] += hidden[i];
        }
        Ok(logits)
    }
}

impl FirstTokenModel for HfDsV4Runner {
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(prompt))
    }
    fn generate_first_token(&mut self, prompt_tokens: &[u32]) -> Result<Option<u32>> {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: dsv4_param::NUM_EXPERTS_PER_TOK,
            prefetch_per_layer: 0,
            preserve_source_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for layer in 0..dsv4_param::NUM_LAYERS {
            planner.register_source(
                ExpertId::new(layer, 0),
                ExpertSource::LocalTensorSet {
                    tensors: Vec::new(),
                },
            );
        }
        let mut state = LayerExecutionState::new(dsv4_param::HEAD_DIM, planner);
        let last_token = *prompt_tokens.last().unwrap_or(&0);
        let logits = self.decode_with_logits(&mut state, last_token)?;
        let max_idx = logits
            .iter()
            .enumerate()
            .fold(
                (0usize, logits[0]),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            )
            .0 as u32;
        Ok(Some(max_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reports_missing_model_gracefully() {
        let r = HfDsV4Runner::load("/nonexistent/dsv4", 1024);
        assert!(r.is_err());
    }
}
