//! Algorithmic DeepSeek-V4 checkpoint-name mapping.

use crate::nn::{ModulePath, ParameterPart};
use crate::transformer::{
    ExternalTensorMeta, NameMapError, NameMapper, NameMapping, TensorTransform,
};

use super::DeepSeekV4Config;

/// Maps native DeepSeek-V4/Hugging Face tensor names to recipe canonical paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeepSeekV4NameMapper {
    layers: usize,
    experts: usize,
    mtp_stages: usize,
}

impl DeepSeekV4NameMapper {
    fn new(layers: usize, experts: usize) -> Result<Self, NameMapError> {
        Self::with_mtp_stages(layers, experts, 0)
    }

    fn with_mtp_stages(
        layers: usize,
        experts: usize,
        mtp_stages: usize,
    ) -> Result<Self, NameMapError> {
        if layers == 0 || experts == 0 {
            return Err(NameMapError::new(
                "DeepSeek-V4 mapper requires non-zero layer and expert counts",
            ));
        }
        Ok(Self {
            layers,
            experts,
            mtp_stages,
        })
    }

    pub(super) fn from_config(config: &DeepSeekV4Config) -> Result<Self, NameMapError> {
        Self::new(config.num_layers, config.num_routed_experts)
    }

    pub(super) fn from_config_with_mtp_stages(
        config: &DeepSeekV4Config,
        mtp_stages: usize,
    ) -> Result<Self, NameMapError> {
        Self::with_mtp_stages(config.num_layers, config.num_routed_experts, mtp_stages)
    }

    fn top_level_mapping(&self, name: &str) -> Result<Option<NameMapping>, NameMapError> {
        let canonical = match name {
            "embed.weight"
            | "model.embed_tokens.weight"
            | "tok_embeddings.weight"
            | "token_embd.weight" => "token_embedding.weight",
            "norm.weight" | "model.norm.weight" | "output_norm.weight" => "final_norm.weight",
            "head.weight" | "lm_head.weight" | "output.weight" => "output.weight",
            "hc_head_fn" => "output_hyper_connection.function",
            "hc_head_scale" => "output_hyper_connection.scale",
            "hc_head_base" => "output_hyper_connection.base",
            _ => return Ok(None),
        };
        mapping(canonical, ParameterPart::Weight)
    }

    fn layer_mapping(&self, name: &str) -> Result<Option<NameMapping>, NameMapError> {
        let native = name.strip_prefix("model.").unwrap_or(name);
        let segments = native.split('.').collect::<Vec<_>>();
        let ["layers", layer, tail @ ..] = segments.as_slice() else {
            return invalid(name);
        };
        let layer = parse_index(layer, "layer", name)?;
        if layer >= self.layers {
            return Err(NameMapError::new(format!(
                "DeepSeek-V4 layer {layer} is outside 0..{} in '{name}'",
                self.layers
            )));
        }
        self.block_mapping(name, &format!("layers.{layer}"), tail, Some(layer), false)
    }

    fn mtp_mapping(&self, name: &str) -> Result<Option<NameMapping>, NameMapError> {
        let segments = name.split('.').collect::<Vec<_>>();
        let ["mtp", stage, tail @ ..] = segments.as_slice() else {
            return invalid(name);
        };
        let stage = parse_index(stage, "MTP stage", name)?;
        if stage >= self.mtp_stages {
            return Err(NameMapError::new(format!(
                "DeepSeek-V4 MTP stage {stage} is outside 0..{} in '{name}'",
                self.mtp_stages
            )));
        }
        let prefix = format!("attachments.mtp.{stage}");
        match tail {
            ["main_proj", part] => {
                if stage != 0 {
                    return invalid(name);
                }
                matrix_mapping(name, format!("{prefix}.main_projection.weight"), part)
            }
            ["main_norm", "weight"] => {
                if stage != 0 {
                    return invalid(name);
                }
                mapping(format!("{prefix}.main_norm.weight"), ParameterPart::Weight)
            }
            [field] if field.starts_with("hc_head_") => {
                if stage + 1 != self.mtp_stages {
                    return invalid(name);
                }
                let component = hc_component(field, "hc_head_", name)?;
                mapping(
                    format!("{prefix}.heads.hyper_connection.{component}"),
                    ParameterPart::Weight,
                )
            }
            ["norm", "weight"] => {
                if stage + 1 != self.mtp_stages {
                    return invalid(name);
                }
                mapping(format!("{prefix}.heads.norm.weight"), ParameterPart::Weight)
            }
            ["markov_head", "markov_w1", part] => {
                if stage + 1 != self.mtp_stages {
                    return invalid(name);
                }
                matrix_mapping(
                    name,
                    format!("{prefix}.heads.markov_embedding.weight"),
                    part,
                )
            }
            ["markov_head", "markov_w2", part] => {
                if stage + 1 != self.mtp_stages {
                    return invalid(name);
                }
                matrix_mapping(name, format!("{prefix}.heads.markov_output.weight"), part)
            }
            ["confidence_head", "proj", part] => {
                if stage + 1 != self.mtp_stages {
                    return invalid(name);
                }
                matrix_mapping(name, format!("{prefix}.heads.confidence.weight"), part)
            }
            _ => self.block_mapping(name, &prefix, tail, None, true),
        }
    }

    fn block_mapping(
        &self,
        name: &str,
        prefix: &str,
        tail: &[&str],
        layer: Option<usize>,
        mtp: bool,
    ) -> Result<Option<NameMapping>, NameMapError> {
        match tail {
            ["attn_norm", "weight"] | ["input_layernorm", "weight"] => {
                mapping(format!("{prefix}.input_norm.weight"), ParameterPart::Weight)
            }
            ["ffn_norm", "weight"] | ["post_attention_layernorm", "weight"] => mapping(
                format!("{prefix}.post_attention_norm.weight"),
                ParameterPart::Weight,
            ),
            [field] if field.starts_with("hc_attn_") => {
                let component = hc_component(field, "hc_attn_", name)?;
                mapping(
                    format!("{prefix}.hyper_connection.attention.{component}"),
                    ParameterPart::Weight,
                )
            }
            [field] if field.starts_with("hc_ffn_") => {
                let component = hc_component(field, "hc_ffn_", name)?;
                mapping(
                    format!("{prefix}.hyper_connection.feed_forward.{component}"),
                    ParameterPart::Weight,
                )
            }
            ["attn", attention @ ..] => self.attention_mapping(name, prefix, attention, mtp),
            ["ffn", ffn @ ..] => self.ffn_mapping(name, prefix, ffn, layer),
            _ => invalid(name),
        }
    }

    fn attention_mapping(
        &self,
        name: &str,
        prefix: &str,
        tail: &[&str],
        mtp: bool,
    ) -> Result<Option<NameMapping>, NameMapError> {
        let attention = format!("{prefix}.attention");
        match tail {
            [field, part] if matches!(*field, "wq_a" | "wq_b" | "wkv" | "wo_a" | "wo_b") => {
                let component = match *field {
                    "wq_a" => "query_a",
                    "wq_b" => "query_b",
                    "wkv" => "key_value",
                    "wo_a" => "output_a",
                    "wo_b" => "output_b",
                    _ => unreachable!(),
                };
                matrix_mapping(name, format!("{attention}.{component}.weight"), part)
            }
            ["q_norm", "weight"] => mapping(
                format!("{attention}.query_norm.weight"),
                ParameterPart::Weight,
            ),
            ["kv_norm", "weight"] => mapping(
                format!("{attention}.key_value_norm.weight"),
                ParameterPart::Weight,
            ),
            ["attn_sink"] | ["attn_sink", "weight"] => {
                mapping(format!("{attention}.sink"), ParameterPart::Weight)
            }
            ["compressor", compressor @ ..] => {
                if mtp {
                    return invalid(name);
                }
                compressor_mapping(name, &format!("{attention}.compressor"), compressor)
            }
            ["indexer", "compressor", compressor @ ..] => {
                if mtp {
                    return invalid(name);
                }
                compressor_mapping(name, &format!("{attention}.indexer.compressor"), compressor)
            }
            ["indexer", "wq_b", part] => {
                if mtp {
                    return invalid(name);
                }
                matrix_mapping(name, format!("{attention}.indexer.query.weight"), part)
            }
            ["indexer", "weights_proj", "weight"] => {
                if mtp {
                    return invalid(name);
                }
                mapping(
                    format!("{attention}.indexer.weights.weight"),
                    ParameterPart::Weight,
                )
            }
            _ => invalid(name),
        }
    }

    fn ffn_mapping(
        &self,
        name: &str,
        prefix: &str,
        tail: &[&str],
        layer: Option<usize>,
    ) -> Result<Option<NameMapping>, NameMapError> {
        match tail {
            ["gate", "weight"] => mapping(format!("{prefix}.router.weight"), ParameterPart::Weight),
            ["gate", "bias"] => mapping(format!("{prefix}.router.bias"), ParameterPart::Weight),
            ["gate", "tid2eid"] => {
                if layer.is_none() {
                    return invalid(name);
                }
                mapping(format!("{prefix}.router.hash"), ParameterPart::Weight)
            }
            ["shared_experts", matrix, part] => {
                let component = expert_component(matrix, name)?;
                matrix_mapping(
                    name,
                    format!("{prefix}.shared_expert.{component}.weight"),
                    part,
                )
            }
            ["experts", expert, matrix, part] => {
                let expert = parse_index(expert, "expert", name)?;
                if expert >= self.experts {
                    return Err(NameMapError::new(format!(
                        "DeepSeek-V4 expert {expert} is outside 0..{} in '{name}'",
                        self.experts
                    )));
                }
                let component = expert_component(matrix, name)?;
                matrix_mapping(
                    name,
                    format!("{prefix}.experts.{expert}.{component}.weight"),
                    part,
                )
            }
            _ => invalid(name),
        }
    }
}

impl NameMapper for DeepSeekV4NameMapper {
    fn map(
        &self,
        external_name: &str,
        _meta: ExternalTensorMeta<'_>,
    ) -> Result<Option<NameMapping>, NameMapError> {
        if let Some(mapping) = self.top_level_mapping(external_name)? {
            return Ok(Some(mapping));
        }
        if external_name.starts_with("layers.") || external_name.starts_with("model.layers.") {
            return self.layer_mapping(external_name);
        }
        if external_name.starts_with("mtp.") {
            return self.mtp_mapping(external_name);
        }
        if looks_deepseek_like(external_name) {
            return invalid(external_name);
        }
        Ok(None)
    }
}

fn compressor_mapping(
    name: &str,
    prefix: &str,
    tail: &[&str],
) -> Result<Option<NameMapping>, NameMapError> {
    let canonical = match tail {
        ["ape"] | ["ape", "weight"] => format!("{prefix}.ape"),
        ["norm", "weight"] => format!("{prefix}.norm.weight"),
        ["wkv", "weight"] => format!("{prefix}.key_value.weight"),
        ["wgate", "weight"] => format!("{prefix}.gate.weight"),
        _ => return invalid(name),
    };
    mapping(canonical, ParameterPart::Weight)
}

fn matrix_mapping(
    name: &str,
    canonical: String,
    part: &str,
) -> Result<Option<NameMapping>, NameMapError> {
    let part = match part {
        "weight" => ParameterPart::Weight,
        "scale" => ParameterPart::Scale,
        _ => return invalid(name),
    };
    mapping(canonical, part)
}

fn expert_component(matrix: &str, name: &str) -> Result<&'static str, NameMapError> {
    match matrix {
        "w1" => Ok("gate"),
        "w3" => Ok("up"),
        "w2" => Ok("down"),
        _ => Err(NameMapError::new(format!(
            "unknown or malformed DeepSeek-V4 tensor name '{name}'"
        ))),
    }
}

fn hc_component(field: &str, prefix: &str, name: &str) -> Result<&'static str, NameMapError> {
    match field.strip_prefix(prefix) {
        Some("fn") => Ok("function"),
        Some("scale") => Ok("scale"),
        Some("base") => Ok("base"),
        _ => Err(NameMapError::new(format!(
            "unknown or malformed DeepSeek-V4 tensor name '{name}'"
        ))),
    }
}

pub(super) fn mapping(
    canonical: impl Into<String>,
    part: ParameterPart,
) -> Result<Option<NameMapping>, NameMapError> {
    let path = ModulePath::new(canonical.into()).map_err(NameMapError::invalid_canonical_path)?;
    Ok(Some(NameMapping::new(
        path,
        part,
        TensorTransform::Identity,
    )))
}

fn parse_index(value: &str, kind: &str, name: &str) -> Result<usize, NameMapError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(NameMapError::new(format!(
            "invalid DeepSeek-V4 {kind} index in '{name}'"
        )));
    }
    value.parse().map_err(|_| {
        NameMapError::new(format!(
            "DeepSeek-V4 {kind} index overflows usize in '{name}'"
        ))
    })
}

fn looks_deepseek_like(name: &str) -> bool {
    [
        "embed.",
        "model.embed_tokens.",
        "tok_embeddings.",
        "token_embd.",
        "norm.",
        "model.norm.",
        "output_norm.",
        "head.",
        "lm_head.",
        "output.",
        "hc_head_",
        "model.layers.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn invalid<T>(name: &str) -> Result<T, NameMapError> {
    Err(NameMapError::new(format!(
        "unknown or malformed DeepSeek-V4 tensor name '{name}'"
    )))
}
