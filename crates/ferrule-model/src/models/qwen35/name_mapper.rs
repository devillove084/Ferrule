//! Exact profile-derived naming and header validation, independent of decoder IR.

use std::collections::{BTreeMap, BTreeSet};

use ferrule_common::Result;

use crate::HfSafetensorsInventory;
use crate::nn::{ModulePath, ParameterDType, ParameterPart};
use crate::transformer::{ExternalTensorMeta, NameMapError, NameMapper, NameMapping};

use super::{Qwen35Config, Qwen35LayerType, model_error};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35TensorPartitionKind {
    Text,
    Visual,
    Mtp,
}

/// Physical storage requirement. Attachments have no text canonical path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TensorSpec {
    pub external_name: String,
    pub canonical_path: Option<ModulePath>,
    pub partition: Qwen35TensorPartitionKind,
    pub part: ParameterPart,
    pub dtype: ParameterDType,
    pub shape: Vec<usize>,
}

impl Qwen35TensorSpec {
    pub fn bytes(&self) -> u64 {
        // Only constructed from a validated, bounded profile.
        self.shape.iter().product::<usize>() as u64
            * self
                .dtype
                .element_size_bytes()
                .expect("known profile dtype") as u64
    }
}

/// Indices into the inventory passed to validate_inventory, not payload copies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35TensorPartition {
    text: Vec<usize>,
    visual: Vec<usize>,
    mtp: Vec<usize>,
}

impl Qwen35TensorPartition {
    pub fn text(&self) -> &[usize] {
        &self.text
    }
    pub fn visual(&self) -> &[usize] {
        &self.visual
    }
    pub fn mtp(&self) -> &[usize] {
        &self.mtp
    }
}

#[derive(Debug, Clone)]
pub struct Qwen35HfNameMapper {
    tensors: BTreeMap<String, Qwen35TensorSpec>,
    tied_output: bool,
}

impl Qwen35HfNameMapper {
    pub fn new(config: &Qwen35Config) -> Self {
        let mut mapper = Self {
            tensors: BTreeMap::new(),
            tied_output: config.text().tie_word_embeddings,
        };
        let t = config.text();
        mapper.text(
            "model.language_model.embed_tokens.weight",
            "token_embedding.weight",
            &[t.vocab_size, t.hidden_size],
            ParameterDType::Bf16,
        );
        mapper.text(
            "model.language_model.norm.weight",
            "final_norm.weight",
            &[t.hidden_size],
            ParameterDType::Bf16,
        );
        if !t.tie_word_embeddings {
            mapper.text(
                "lm_head.weight",
                "output.weight",
                &[t.vocab_size, t.hidden_size],
                ParameterDType::Bf16,
            );
        }
        for (i, kind) in config.layer_types().iter().enumerate() {
            let external = format!("model.language_model.layers.{i}");
            let canonical = format!("layers.{i}");
            for (suffix, target, shape) in common_layer(config) {
                mapper.text(
                    &format!("{external}.{suffix}"),
                    &format!("{canonical}.{target}"),
                    &shape,
                    ParameterDType::Bf16,
                );
            }
            match kind {
                Qwen35LayerType::FullAttention => {
                    for (suffix, target, shape) in full_attention(config) {
                        mapper.text(
                            &format!("{external}.self_attn.{suffix}"),
                            &format!("{canonical}.attention.{target}"),
                            &shape,
                            ParameterDType::Bf16,
                        );
                    }
                }
                Qwen35LayerType::LinearAttention => {
                    let keys = t.linear_num_key_heads * t.linear_key_head_dim;
                    let values = t.linear_num_value_heads * t.linear_value_head_dim;
                    for (suffix, target, shape, dtype) in [
                        (
                            "in_proj_qkv.weight",
                            "query_key_value.weight",
                            vec![2 * keys + values, t.hidden_size],
                            ParameterDType::Bf16,
                        ),
                        (
                            "in_proj_z.weight",
                            "gate.weight",
                            vec![values, t.hidden_size],
                            ParameterDType::Bf16,
                        ),
                        (
                            "in_proj_b.weight",
                            "beta.weight",
                            vec![t.linear_num_value_heads, t.hidden_size],
                            ParameterDType::Bf16,
                        ),
                        (
                            "in_proj_a.weight",
                            "decay.weight",
                            vec![t.linear_num_value_heads, t.hidden_size],
                            ParameterDType::Bf16,
                        ),
                        (
                            "conv1d.weight",
                            "convolution.weight",
                            vec![2 * keys + values, 1, t.linear_conv_kernel_dim],
                            ParameterDType::Bf16,
                        ),
                        (
                            "dt_bias",
                            "time_bias.weight",
                            vec![t.linear_num_value_heads],
                            ParameterDType::Bf16,
                        ),
                        (
                            "A_log",
                            "decay_log.weight",
                            vec![t.linear_num_value_heads],
                            ParameterDType::F32,
                        ),
                        (
                            "norm.weight",
                            "norm.weight",
                            vec![t.linear_value_head_dim],
                            ParameterDType::F32,
                        ),
                        (
                            "out_proj.weight",
                            "output.weight",
                            vec![t.hidden_size, values],
                            ParameterDType::Bf16,
                        ),
                    ] {
                        mapper.text(
                            &format!("{external}.linear_attn.{suffix}"),
                            &format!("{canonical}.linear_attention.{target}"),
                            &shape,
                            dtype,
                        );
                    }
                }
            }
        }
        mapper.visual(config);
        mapper.mtp(config);
        if config.numeric_fp8_encoding().is_some() {
            mapper.apply_numeric_fp8();
        }
        mapper
    }

    pub fn tensors(&self) -> impl Iterator<Item = &Qwen35TensorSpec> {
        self.tensors.values()
    }

    pub const fn tied_output(&self) -> bool {
        self.tied_output
    }

    /// Dense aliases embedding storage; MoE requires physical `lm_head.weight`.
    pub const fn output_alias(&self) -> Option<(&'static str, &'static str)> {
        if self.tied_output {
            Some(("output.weight", "token_embedding.weight"))
        } else {
            None
        }
    }

    pub fn validate_tensor(
        &self,
        name: &str,
        meta: ExternalTensorMeta<'_>,
    ) -> std::result::Result<&Qwen35TensorSpec, NameMapError> {
        let spec = self.tensors.get(name).ok_or_else(|| {
            NameMapError::new(format!(
                "Qwen3.5: unknown or unsupported tensor '{name}' (not in the exact profile schema)"
            ))
        })?;
        if meta.dtype != spec.dtype.as_str()
            || meta.shape != spec.shape
            || meta.bytes != spec.bytes()
        {
            return Err(NameMapError::new(format!(
                "Qwen3.5 tensor '{name}' requires {} {:?} / {} bytes, got {} {:?} / {} bytes",
                spec.dtype.as_str(),
                spec.shape,
                spec.bytes(),
                meta.dtype,
                meta.shape,
                meta.bytes
            )));
        }
        Ok(spec)
    }

    /// Require all text tensors. Each optional attachment is either absent or
    /// complete; accepting a subtree prefix alone is never sufficient.
    pub fn validate_inventory(
        &self,
        inventory: &HfSafetensorsInventory,
    ) -> Result<Qwen35TensorPartition> {
        if !inventory.index_only_tensors.is_empty() || !inventory.header_only_tensors.is_empty() {
            return Err(model_error("safetensors index/header tensor sets differ"));
        }
        if inventory.tensor_count != inventory.tensors.len() {
            return Err(model_error(
                "inventory tensor_count does not match tensor entries",
            ));
        }
        let mut result = Qwen35TensorPartition {
            text: Vec::new(),
            visual: Vec::new(),
            mtp: Vec::new(),
        };
        let mut seen = BTreeSet::new();
        for (index, tensor) in inventory.tensors.iter().enumerate() {
            if !seen.insert(tensor.name.as_str()) {
                return Err(model_error(format!("duplicate tensor '{}'", tensor.name)));
            }
            let spec = self
                .validate_tensor(
                    &tensor.name,
                    ExternalTensorMeta {
                        dtype: &tensor.dtype,
                        shape: &tensor.shape,
                        bytes: tensor.byte_size,
                    },
                )
                .map_err(|error| model_error(error.to_string()))?;
            match spec.partition {
                Qwen35TensorPartitionKind::Text => result.text.push(index),
                Qwen35TensorPartitionKind::Visual => result.visual.push(index),
                Qwen35TensorPartitionKind::Mtp => result.mtp.push(index),
            }
        }
        for spec in self.tensors.values() {
            let required = match spec.partition {
                Qwen35TensorPartitionKind::Text => true,
                Qwen35TensorPartitionKind::Visual => !result.visual.is_empty(),
                Qwen35TensorPartitionKind::Mtp => !result.mtp.is_empty(),
            };
            if required && !seen.contains(spec.external_name.as_str()) {
                return Err(model_error(format!(
                    "incomplete {:?} partition: missing '{}'",
                    spec.partition, spec.external_name
                )));
            }
        }
        Ok(result)
    }

    fn text(&mut self, name: &str, canonical: &str, shape: &[usize], dtype: ParameterDType) {
        self.insert(
            name,
            Some(ModulePath::new(canonical).expect("static canonical name")),
            Qwen35TensorPartitionKind::Text,
            shape,
            dtype,
        );
    }

    fn attachment(&mut self, name: &str, shape: &[usize], partition: Qwen35TensorPartitionKind) {
        self.insert(name, None, partition, shape, ParameterDType::Bf16);
    }

    fn insert(
        &mut self,
        name: &str,
        canonical_path: Option<ModulePath>,
        partition: Qwen35TensorPartitionKind,
        shape: &[usize],
        dtype: ParameterDType,
    ) {
        let previous = self.tensors.insert(
            name.to_owned(),
            Qwen35TensorSpec {
                external_name: name.to_owned(),
                canonical_path,
                partition,
                part: ParameterPart::Weight,
                dtype,
                shape: shape.to_vec(),
            },
        );
        assert!(previous.is_none(), "duplicate static Qwen3.5 schema name");
    }

    fn apply_numeric_fp8(&mut self) {
        let mut scales = Vec::new();
        for tensor in self.tensors.values_mut() {
            let name = &tensor.external_name;
            let quantized = name.contains(".mlp.experts.")
                || name.contains(".mlp.shared_expert.")
                || [
                    ".self_attn.q_proj.weight",
                    ".self_attn.k_proj.weight",
                    ".self_attn.v_proj.weight",
                    ".self_attn.o_proj.weight",
                    ".linear_attn.in_proj_qkv.weight",
                    ".linear_attn.in_proj_z.weight",
                    ".linear_attn.out_proj.weight",
                ]
                .iter()
                .any(|suffix| name.ends_with(suffix));
            if quantized {
                assert_eq!(tensor.shape.len(), 2);
                tensor.dtype = ParameterDType::F8E4M3;
                let mut scale = tensor.clone();
                scale.external_name = format!("{}_scale_inv", tensor.external_name);
                scale.dtype = ParameterDType::Bf16;
                scale.part = ParameterPart::Scale;
                scale.shape = tensor.shape.iter().map(|n| n.div_ceil(128)).collect();
                scales.push(scale);
            }
        }
        for scale in scales {
            assert!(
                self.tensors
                    .insert(scale.external_name.clone(), scale)
                    .is_none()
            );
        }
    }

    fn visual(&mut self, config: &Qwen35Config) {
        let v = config.vision();
        let h = v.hidden_size;
        let m = v.intermediate_size;
        let merged = h * v.spatial_merge_size * v.spatial_merge_size;
        let mut add = |name: &str, shape: &[usize]| {
            self.attachment(
                &format!("model.visual.{name}"),
                shape,
                Qwen35TensorPartitionKind::Visual,
            )
        };
        add(
            "patch_embed.proj.weight",
            &[
                h,
                v.in_channels,
                v.temporal_patch_size,
                v.patch_size,
                v.patch_size,
            ],
        );
        add("patch_embed.proj.bias", &[h]);
        add("pos_embed.weight", &[v.num_position_embeddings, h]);
        for (suffix, shape) in [
            ("norm.weight", vec![h]),
            ("norm.bias", vec![h]),
            ("linear_fc1.weight", vec![merged, merged]),
            ("linear_fc1.bias", vec![merged]),
            ("linear_fc2.weight", vec![v.out_hidden_size, merged]),
            ("linear_fc2.bias", vec![v.out_hidden_size]),
        ] {
            add(&format!("merger.{suffix}"), &shape);
        }
        for i in 0..v.depth {
            for (suffix, shape) in [
                ("norm1.weight", vec![h]),
                ("norm1.bias", vec![h]),
                ("norm2.weight", vec![h]),
                ("norm2.bias", vec![h]),
                ("attn.qkv.weight", vec![3 * h, h]),
                ("attn.qkv.bias", vec![3 * h]),
                ("attn.proj.weight", vec![h, h]),
                ("attn.proj.bias", vec![h]),
                ("mlp.linear_fc1.weight", vec![m, h]),
                ("mlp.linear_fc1.bias", vec![m]),
                ("mlp.linear_fc2.weight", vec![h, m]),
                ("mlp.linear_fc2.bias", vec![h]),
            ] {
                add(&format!("blocks.{i}.{suffix}"), &shape);
            }
        }
    }

    fn mtp(&mut self, config: &Qwen35Config) {
        let t = config.text();
        let mut add = |name: &str, shape: &[usize]| {
            self.attachment(
                &format!("mtp.{name}"),
                shape,
                Qwen35TensorPartitionKind::Mtp,
            )
        };
        add("fc.weight", &[t.hidden_size, 2 * t.hidden_size]);
        for suffix in [
            "norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        ] {
            add(suffix, &[t.hidden_size]);
        }
        for i in 0..t.mtp_num_hidden_layers {
            for (suffix, _, shape) in common_layer(config) {
                add(&format!("layers.{i}.{suffix}"), &shape);
            }
            for (suffix, _, shape) in full_attention(config) {
                add(&format!("layers.{i}.self_attn.{suffix}"), &shape);
            }
        }
    }
}

impl NameMapper for Qwen35HfNameMapper {
    fn map(
        &self,
        name: &str,
        meta: ExternalTensorMeta<'_>,
    ) -> std::result::Result<Option<NameMapping>, NameMapError> {
        let spec = self.validate_tensor(name, meta)?;
        let path = spec.canonical_path.clone().ok_or_else(|| NameMapError::new(format!(
            "Qwen3.5 validated {:?} attachment '{name}' must be partitioned before text state-dict binding", spec.partition
        )))?;
        Ok(Some(match spec.part {
            ParameterPart::Weight => NameMapping::weight(path),
            ParameterPart::Scale => NameMapping::scale(path),
        }))
    }
}

fn common_layer(config: &Qwen35Config) -> Vec<(String, String, Vec<usize>)> {
    let t = config.text();
    let h = t.hidden_size;
    let mut tensors = vec![
        (
            "input_layernorm.weight".into(),
            "input_norm.weight".into(),
            vec![h],
        ),
        (
            "post_attention_layernorm.weight".into(),
            "post_attention_norm.weight".into(),
            vec![h],
        ),
    ];
    let mut swiglu = |external: &str, canonical: &str, intermediate: usize| {
        for (projection, shape) in [
            ("gate", vec![intermediate, h]),
            ("up", vec![intermediate, h]),
            ("down", vec![h, intermediate]),
        ] {
            tensors.push((
                format!("{external}.{projection}_proj.weight"),
                format!("{canonical}.{projection}.weight"),
                shape,
            ));
        }
    };
    if let Some(experts) = t.num_experts {
        for expert in 0..experts {
            swiglu(
                &format!("mlp.experts.{expert}"),
                &format!("feed_forward.experts.{expert}"),
                t.moe_intermediate_size.expect("validated MoE"),
            );
        }
        swiglu(
            "mlp.shared_expert",
            "feed_forward.shared_expert",
            t.shared_expert_intermediate_size
                .expect("validated shared expert"),
        );
        tensors.push((
            "mlp.gate.weight".into(),
            "feed_forward.router.weight".into(),
            vec![experts, h],
        ));
        tensors.push((
            "mlp.shared_expert_gate.weight".into(),
            "feed_forward.shared_expert_gate.weight".into(),
            vec![1, h],
        ));
    } else {
        swiglu("mlp", "feed_forward", t.intermediate_size);
    }
    tensors
}

fn full_attention(config: &Qwen35Config) -> Vec<(&'static str, &'static str, Vec<usize>)> {
    let t = config.text();
    let q = t.num_attention_heads * t.head_dim;
    let kv = t.num_key_value_heads * t.head_dim;
    vec![
        (
            "q_proj.weight",
            "query_gate.weight",
            vec![2 * q, t.hidden_size],
        ),
        ("k_proj.weight", "key.weight", vec![kv, t.hidden_size]),
        ("v_proj.weight", "value.weight", vec![kv, t.hidden_size]),
        ("o_proj.weight", "output.weight", vec![t.hidden_size, q]),
        ("q_norm.weight", "query_norm.weight", vec![t.head_dim]),
        ("k_norm.weight", "key_norm.weight", vec![t.head_dim]),
    ]
}
