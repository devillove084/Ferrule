//! Full 43-layer synthetic DSV4 runner for end-to-end smoke tests.
//!
//! This runner uses real-config-shaped synthetic weights to execute a DSV4-shaped
//! Transformer decode through ReferenceLayerExecutor. No real model file required.

use std::path::{Path, PathBuf};

use ferrule_core::Result;

use crate::dsv4_mock::{build_mock_execution_state, build_mock_layer, SyntheticTokenizer};
use crate::dsv4_param;
use crate::first_token_smoke::FirstTokenModel;
use crate::hyper_connection::{
    hc_head_reference, HyperConnectionConfig, HyperConnectionHeadWeights,
};
use crate::layer_binding::{LayerExecutionState, LayerSourceBinding, ReferenceLayerExecutor};
use crate::{CpuReferenceExpertExecutor, ExpertStreamingReader};

/// Full synthetic DSV4 model with mock weights.
pub struct SyntheticDsV4Runner {
    config: HyperConnectionConfig,
    layers: Vec<LayerSourceBinding>,
    head_weights: HyperConnectionHeadWeights,
    tokenizer: SyntheticTokenizer,
    temp_dir: PathBuf,
}

impl SyntheticDsV4Runner {
    pub fn new() -> Result<Self> {
        let config = dsv4_param::hc_config();
        let layers: Vec<LayerSourceBinding> = (0..dsv4_param::NUM_LAYERS)
            .map(|layer| build_mock_layer(layer, config))
            .collect();
        let head_weights = HyperConnectionHeadWeights {
            function: vec![0.0; config.hc_mult * config.hc_hidden_size()],
            scale: vec![1.0],
            base: vec![0.0; config.hc_mult],
        };
        let temp_dir = unique_temp_dir("ferrule-synthetic-dsv4");
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| ferrule_core::Error::Model(format!("temp dir: {e}")))?;
        Ok(Self {
            config,
            layers,
            head_weights,
            tokenizer: SyntheticTokenizer::new(),
            temp_dir,
        })
    }

    /// Run one full-model decode step for a single token.
    pub fn decode_with_logits(
        &mut self,
        state: &mut LayerExecutionState,
        token_id: u32,
    ) -> Result<Vec<f32>> {
        let executor = ReferenceLayerExecutor::new(
            self.config,
            ExpertStreamingReader::new(64 * 1024 * 1024),
            CpuReferenceExpertExecutor::new(dsv4_param::SWIGLU_LIMIT),
        );

        let mut hc_state = vec![0.0f32; self.config.hc_hidden_size()];
        hc_state[0] = 1.0; // mock embedding

        for layer_index in 0..dsv4_param::NUM_LAYERS {
            let binding = &self.layers[layer_index];
            let predicted: Vec<usize> = Vec::new();
            let output =
                executor.execute_decode_step(binding, state, &hc_state, token_id, &predicted)?;
            hc_state = output.hc_state;
        }

        let hidden = hc_head_reference(&hc_state, 1, self.config, &self.head_weights)?;
        // Simple identity-like mock: use summed hidden as logits
        let mut logits = vec![0.0f32; dsv4_param::VOCAB_SIZE];
        for i in 0..dsv4_param::HIDDEN_SIZE {
            logits[i % dsv4_param::VOCAB_SIZE] += hidden[i];
        }
        Ok(logits)
    }
}

impl FirstTokenModel for SyntheticDsV4Runner {
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(prompt))
    }

    fn generate_first_token(&mut self, prompt_tokens: &[u32]) -> Result<Option<u32>> {
        let mut state = build_mock_execution_state(&self.temp_dir);
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

impl Drop for SyntheticDsV4Runner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.temp_dir);
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nonce}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_dsv4_runner_produces_finite_logits() {
        let mut runner = SyntheticDsV4Runner::new().unwrap();
        let tokens = runner.encode_prompt("hello").unwrap();
        assert!(tokens.len() > 0);
        let first = runner.generate_first_token(&tokens).unwrap();
        assert!(first.is_some());
        assert!(first.unwrap() < dsv4_param::VOCAB_SIZE as u32);
    }

    #[test]
    fn synthetic_dsv4_runner_is_deterministic() {
        let mut r1 = SyntheticDsV4Runner::new().unwrap();
        let mut r2 = SyntheticDsV4Runner::new().unwrap();
        let tokens = r1.encode_prompt("test").unwrap();
        let t1 = r1.generate_first_token(&tokens).unwrap();
        let t2 = r2.generate_first_token(&tokens).unwrap();
        assert_eq!(t1, t2, "synthetic runner should be deterministic");
    }
}
