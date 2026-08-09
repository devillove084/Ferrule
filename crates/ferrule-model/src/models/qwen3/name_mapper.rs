//! Algorithmic Hugging Face to canonical Qwen3-MoE name mapping.

use crate::nn::ModulePath;
use crate::transformer::{ExternalTensorMeta, NameMapError, NameMapper, NameMapping};

use super::Qwen3MoeConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen3HfNameMapper {
    layers: usize,
    experts: usize,
    tied_head: bool,
}

impl Qwen3HfNameMapper {
    fn new(layers: usize, experts: usize, tied_head: bool) -> Result<Self, NameMapError> {
        if layers == 0 || experts == 0 {
            return Err(NameMapError::new(
                "Qwen3 mapper requires non-zero layer and expert counts",
            ));
        }
        Ok(Self {
            layers,
            experts,
            tied_head,
        })
    }

    pub(super) fn from_config(config: &Qwen3MoeConfig) -> Result<Self, NameMapError> {
        Self::new(
            config.num_hidden_layers,
            config.num_experts,
            config.tie_word_embeddings,
        )
    }

    pub(super) fn mapping(path: String) -> Result<Option<NameMapping>, NameMapError> {
        Ok(Some(NameMapping::weight(
            ModulePath::new(path).map_err(NameMapError::invalid_canonical_path)?,
        )))
    }

    fn layer_mapping(&self, name: &str) -> Result<Option<NameMapping>, NameMapError> {
        let segments = name.split('.').collect::<Vec<_>>();
        if segments.len() < 5 || segments[..2] != ["model", "layers"] {
            return Err(NameMapError::new(format!(
                "invalid Qwen3 layer tensor name '{name}'"
            )));
        }
        let layer = parse_index(segments[2], "layer", name)?;
        if layer >= self.layers {
            return Err(NameMapError::new(format!(
                "Qwen3 layer {layer} is outside 0..{} in '{name}'",
                self.layers
            )));
        }
        let canonical = match &segments[3..] {
            ["input_layernorm", "weight"] => format!("layers.{layer}.input_norm.weight"),
            ["post_attention_layernorm", "weight"] => {
                format!("layers.{layer}.post_attention_norm.weight")
            }
            ["self_attn", external, "weight"] => {
                let component = match *external {
                    "q_proj" => "query",
                    "k_proj" => "key",
                    "v_proj" => "value",
                    "o_proj" => "output",
                    "q_norm" => "query_norm",
                    "k_norm" => "key_norm",
                    _ => return invalid(name),
                };
                format!("layers.{layer}.attention.{component}.weight")
            }
            ["mlp", "gate", "weight"] => format!("layers.{layer}.router.weight"),
            ["mlp", "experts", expert, projection, "weight"] => {
                let expert = parse_index(expert, "expert", name)?;
                if expert >= self.experts {
                    return Err(NameMapError::new(format!(
                        "Qwen3 expert {expert} is outside 0..{} in '{name}'",
                        self.experts
                    )));
                }
                let component = match *projection {
                    "gate_proj" => "gate",
                    "up_proj" => "up",
                    "down_proj" => "down",
                    _ => return invalid(name),
                };
                format!("layers.{layer}.experts.{expert}.{component}.weight")
            }
            _ => return invalid(name),
        };
        Self::mapping(canonical)
    }
}

impl NameMapper for Qwen3HfNameMapper {
    fn map(
        &self,
        external_name: &str,
        _meta: ExternalTensorMeta<'_>,
    ) -> Result<Option<NameMapping>, NameMapError> {
        match external_name {
            "model.embed_tokens.weight" => Self::mapping("token_embedding.weight".into()),
            "model.norm.weight" => Self::mapping("final_norm.weight".into()),
            "lm_head.weight" if !self.tied_head => Self::mapping("output.weight".into()),
            "lm_head.weight" => Err(NameMapError::new(
                "tied Qwen3 checkpoint must not contain a separate lm_head.weight",
            )),
            name if name.starts_with("model.layers.") => self.layer_mapping(name),
            name if name.starts_with("model.") || name.starts_with("lm_head.") => invalid(name),
            _ => Ok(None),
        }
    }
}

fn parse_index(value: &str, kind: &str, name: &str) -> Result<usize, NameMapError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(NameMapError::new(format!(
            "invalid Qwen3 {kind} index in '{name}'"
        )));
    }
    value
        .parse()
        .map_err(|_| NameMapError::new(format!("Qwen3 {kind} index overflows usize in '{name}'")))
}

fn invalid<T>(name: &str) -> Result<T, NameMapError> {
    Err(NameMapError::new(format!(
        "unknown or malformed Qwen3 tensor name '{name}'"
    )))
}
