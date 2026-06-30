use ferrule_runtime::ModelInfo;
use std::path::Path;

// ── info ─────────────────────────────────────────────────────────────────────

pub fn cmd_info(model_dir: &str) -> anyhow::Result<()> {
    let model = ferrule_model::OlmoeModel::load(Path::new(model_dir))?;
    let c = &model.config;
    println!(
        "OLMoE: {}d×{}L, {}e top-{}, vocab={}",
        c.hidden_size, c.num_layers, c.num_experts, c.num_experts_per_tok, c.vocab_size
    );
    Ok(())
}

pub fn print_model_info(info: &ModelInfo) {
    println!(
        "OLMoE: {}d×{}L, {}e top-{} ({})",
        info.hidden_size, info.num_layers, info.num_experts, info.num_experts_per_tok, info.backend
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmd_info_nonexistent() {
        let result = cmd_info("/nonexistent/path/to/model");
        assert!(result.is_err());
    }
}
