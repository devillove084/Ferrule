//! Generic source linear handles and CPU reference matvec.
//!
//! Source-bound model bring-up can exercise several formats: BF16/F32 metadata or
//! auxiliary tensors, FP8 E4M3 block-scaled linears, and FP4 packed routed experts.
//! This module keeps those formats behind one typed linear payload so attention,
//! router, shared expert, logits, and future CUDA dispatch all consume the same
//! abstraction.

use ferrule_core::{Error, Result};
use ferrule_model::TensorRole;

use crate::source_format::{
    dequantize_fp4_e2m1_with_e8m0_scales, dequantize_fp8_e4m3fn_with_e8m0_scales,
};
use crate::source_tensor::{SourceDType, SourceTensorPayload};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceLinearFormat {
    F32 {
        out_features: usize,
        in_features: usize,
    },
    Bf16 {
        out_features: usize,
        in_features: usize,
    },
    Fp8E4M3WithE8M0Scale {
        out_features: usize,
        in_features: usize,
        block_m: usize,
        block_k: usize,
    },
    Fp4E2M1PackedWithE8M0Scale {
        out_features: usize,
        in_features: usize,
        block_size: usize,
    },
}

impl SourceLinearFormat {
    pub fn out_features(&self) -> usize {
        match self {
            Self::F32 { out_features, .. }
            | Self::Bf16 { out_features, .. }
            | Self::Fp8E4M3WithE8M0Scale { out_features, .. }
            | Self::Fp4E2M1PackedWithE8M0Scale { out_features, .. } => *out_features,
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::F32 { in_features, .. }
            | Self::Bf16 { in_features, .. }
            | Self::Fp8E4M3WithE8M0Scale { in_features, .. }
            | Self::Fp4E2M1PackedWithE8M0Scale { in_features, .. } => *in_features,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLinearPayload {
    pub role: TensorRole,
    pub weight: SourceTensorPayload,
    pub scale: Option<SourceTensorPayload>,
    pub format: SourceLinearFormat,
}

impl SourceLinearPayload {
    pub fn from_weight_and_scale(
        role: TensorRole,
        weight: SourceTensorPayload,
        scale: Option<SourceTensorPayload>,
    ) -> Result<Self> {
        let format = infer_source_linear_format(&weight, scale.as_ref())?;
        Ok(Self {
            role,
            weight,
            scale,
            format,
        })
    }

    pub fn reference_matvec(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.format.in_features() {
            return Err(Error::Model(format!(
                "source linear {:?} input length mismatch: expected {}, got {}",
                self.role,
                self.format.in_features(),
                input.len()
            )));
        }
        let weights = self.reference_weights_f32()?;
        Ok(matvec_row_major(
            &weights,
            self.format.out_features(),
            self.format.in_features(),
            input,
        ))
    }

    pub fn reference_weights_f32(&self) -> Result<Vec<f32>> {
        match self.format {
            SourceLinearFormat::F32 {
                out_features,
                in_features,
            } => decode_f32_matrix(&self.weight, out_features, in_features),
            SourceLinearFormat::Bf16 {
                out_features,
                in_features,
            } => decode_bf16_matrix(&self.weight, out_features, in_features),
            SourceLinearFormat::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = self.scale.as_ref().ok_or_else(|| {
                    Error::Model(format!(
                        "source linear {:?} FP8 weight is missing E8M0 scale tensor",
                        self.role
                    ))
                })?;
                dequantize_fp8_e4m3fn_with_e8m0_scales(
                    &self.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                    block_m,
                    block_k,
                )
            }
            SourceLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
                block_size,
            } => {
                let scale = self.scale.as_ref().ok_or_else(|| {
                    Error::Model(format!(
                        "source linear {:?} FP4 weight is missing E8M0 scale tensor",
                        self.role
                    ))
                })?;
                dequantize_fp4_e2m1_with_e8m0_scales(
                    &self.weight.bytes,
                    &scale.bytes,
                    out_features,
                    in_features,
                    block_size,
                )
            }
        }
    }
}

fn infer_source_linear_format(
    weight: &SourceTensorPayload,
    scale: Option<&SourceTensorPayload>,
) -> Result<SourceLinearFormat> {
    if weight.slice.shape.len() != 2 {
        return Err(Error::Model(format!(
            "source linear '{}' expects 2D weight shape, got {:?}",
            weight.slice.name, weight.slice.shape
        )));
    }
    let out = weight.slice.shape[0];
    let width = weight.slice.shape[1];
    match weight.slice.dtype {
        SourceDType::F32 => {
            ensure_no_scale(weight, scale)?;
            ensure_byte_len(weight, out, width, 4)?;
            Ok(SourceLinearFormat::F32 {
                out_features: out,
                in_features: width,
            })
        }
        SourceDType::Bf16 => {
            ensure_no_scale(weight, scale)?;
            ensure_byte_len(weight, out, width, 2)?;
            Ok(SourceLinearFormat::Bf16 {
                out_features: out,
                in_features: width,
            })
        }
        SourceDType::F8E4M3 => {
            let scale = scale.ok_or_else(|| {
                Error::Model(format!(
                    "FP8 source linear '{}' requires E8M0 scale tensor",
                    weight.slice.name
                ))
            })?;
            if scale.slice.dtype != SourceDType::F8E8M0 || scale.slice.shape.len() != 2 {
                return Err(Error::Model(format!(
                    "FP8 source linear '{}' expects 2D F8_E8M0 scale, got dtype={} shape={:?}",
                    weight.slice.name,
                    scale.slice.dtype.as_str(),
                    scale.slice.shape
                )));
            }
            ensure_byte_len(weight, out, width, 1)?;
            let block_m =
                infer_fp8_block(out, scale.slice.shape[0], 128, "out", &weight.slice.name)?;
            let block_k =
                infer_fp8_block(width, scale.slice.shape[1], 128, "in", &weight.slice.name)?;
            Ok(SourceLinearFormat::Fp8E4M3WithE8M0Scale {
                out_features: out,
                in_features: width,
                block_m,
                block_k,
            })
        }
        SourceDType::I8 => {
            let scale = scale.ok_or_else(|| {
                Error::Model(format!(
                    "I8 source linear '{}' requires a scale tensor to infer packed FP4",
                    weight.slice.name
                ))
            })?;
            if scale.slice.dtype != SourceDType::F8E8M0 || scale.slice.shape.len() != 2 {
                return Err(Error::Model(format!(
                    "I8/FP4 source linear '{}' expects 2D F8_E8M0 scale, got dtype={} shape={:?}",
                    weight.slice.name,
                    scale.slice.dtype.as_str(),
                    scale.slice.shape
                )));
            }
            ensure_byte_len(weight, out, width, 1)?;
            let in_features = width.checked_mul(2).ok_or_else(|| {
                Error::Model(format!(
                    "source linear '{}' FP4 logical input dimension overflow",
                    weight.slice.name
                ))
            })?;
            let scale_cols = scale.slice.shape[1];
            if scale.slice.shape[0] != out || scale_cols == 0 || in_features % scale_cols != 0 {
                return Err(Error::Model(format!(
                    "I8/FP4 source linear '{}' scale shape {:?} is incompatible with weight shape {:?}",
                    weight.slice.name, scale.slice.shape, weight.slice.shape
                )));
            }
            let block_size = in_features / scale_cols;
            Ok(SourceLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features: out,
                in_features,
                block_size,
            })
        }
        _ => Err(Error::Model(format!(
            "source linear '{}' has unsupported dtype {}",
            weight.slice.name,
            weight.slice.dtype.as_str()
        ))),
    }
}

fn ensure_no_scale(
    weight: &SourceTensorPayload,
    scale: Option<&SourceTensorPayload>,
) -> Result<()> {
    if let Some(scale) = scale {
        return Err(Error::Model(format!(
            "source linear '{}' has unexpected scale tensor '{}' for dtype {}",
            weight.slice.name,
            scale.slice.name,
            weight.slice.dtype.as_str()
        )));
    }
    Ok(())
}

fn ensure_byte_len(
    tensor: &SourceTensorPayload,
    rows: usize,
    cols: usize,
    bytes_per_element: usize,
) -> Result<()> {
    let expected = rows
        .checked_mul(cols)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or_else(|| {
            Error::Model(format!(
                "source tensor '{}' size overflow",
                tensor.slice.name
            ))
        })?;
    if tensor.bytes.len() != expected || tensor.slice.bytes != expected as u64 {
        return Err(Error::Model(format!(
            "source tensor '{}' byte length mismatch: expected {expected}, metadata={}, payload={}",
            tensor.slice.name,
            tensor.slice.bytes,
            tensor.bytes.len()
        )));
    }
    Ok(())
}

fn infer_fp8_block(
    dim: usize,
    scale_dim: usize,
    preferred: usize,
    axis: &str,
    name: &str,
) -> Result<usize> {
    if scale_dim == 0 {
        return Err(Error::Model(format!(
            "FP8 source linear '{name}' has zero scale {axis} dimension"
        )));
    }
    if dim.div_ceil(preferred) == scale_dim {
        return Ok(preferred);
    }
    let block = dim.div_ceil(scale_dim);
    if dim.div_ceil(block) == scale_dim {
        return Ok(block);
    }
    Err(Error::Model(format!(
        "FP8 source linear '{name}' cannot infer {axis} block size from dim={dim}, scale_dim={scale_dim}"
    )))
}

fn decode_f32_matrix(
    tensor: &SourceTensorPayload,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    ensure_byte_len(tensor, out_features, in_features, 4)?;
    Ok(tensor
        .bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_bf16_matrix(
    tensor: &SourceTensorPayload,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    ensure_byte_len(tensor, out_features, in_features, 2)?;
    Ok(tensor
        .bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
            f32::from_bits(bits << 16)
        })
        .collect())
}

fn matvec_row_major(weights: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for row in 0..rows {
        let mut acc = 0.0f32;
        let offset = row * cols;
        for col in 0..cols {
            acc += weights[offset + col] * input[col];
        }
        output[row] = acc;
    }
    output
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::source_tensor::{SourceDType, SourceTensorSlice};

    #[test]
    fn f32_source_linear_matvec() {
        let linear = SourceLinearPayload::from_weight_and_scale(
            TensorRole::AttentionOutput,
            payload(
                "f32.weight",
                SourceDType::F32,
                vec![2, 3],
                f32_bytes(&[1.0, 2.0, 3.0, -1.0, 0.5, 4.0]),
            ),
            None,
        )
        .unwrap();
        assert_eq!(
            linear.format,
            SourceLinearFormat::F32 {
                out_features: 2,
                in_features: 3,
            }
        );
        assert_eq!(
            linear.reference_matvec(&[1.0, 2.0, 3.0]).unwrap(),
            vec![14.0, 12.0]
        );
    }

    #[test]
    fn bf16_source_linear_matvec() {
        let linear = SourceLinearPayload::from_weight_and_scale(
            TensorRole::RouterLogits,
            payload(
                "bf16.weight",
                SourceDType::Bf16,
                vec![1, 3],
                bf16_bytes(&[1.0, -2.0, 0.5]),
            ),
            None,
        )
        .unwrap();
        assert_eq!(
            linear.reference_matvec(&[2.0, 3.0, 4.0]).unwrap(),
            vec![-2.0]
        );
    }

    #[test]
    fn fp8_source_linear_matvec_uses_e8m0_scales() {
        let linear = SourceLinearPayload::from_weight_and_scale(
            TensorRole::AttentionQuery,
            payload(
                "fp8.weight",
                SourceDType::F8E4M3,
                vec![2, 3],
                vec![0x38, 0x40, 0xb8, 0x00, 0x38, 0x38],
            ),
            Some(payload(
                "fp8.scale",
                SourceDType::F8E8M0,
                vec![2, 2],
                vec![127, 128, 126, 127],
            )),
        )
        .unwrap();
        assert_eq!(
            linear.format,
            SourceLinearFormat::Fp8E4M3WithE8M0Scale {
                out_features: 2,
                in_features: 3,
                block_m: 1,
                block_k: 2,
            }
        );
        assert_eq!(
            linear.reference_matvec(&[1.0, 1.0, 1.0]).unwrap(),
            vec![1.0, 1.5]
        );
    }

    #[test]
    fn fp4_source_linear_matvec_uses_packed_layout() {
        let mut weight = vec![0u8; 16];
        weight[0] = 0x42; // row 0: 1.0, 2.0
        let linear = SourceLinearPayload::from_weight_and_scale(
            TensorRole::RoutedExpertGate,
            payload("fp4.weight", SourceDType::I8, vec![1, 16], weight),
            Some(payload(
                "fp4.scale",
                SourceDType::F8E8M0,
                vec![1, 1],
                vec![127],
            )),
        )
        .unwrap();
        assert_eq!(
            linear.format,
            SourceLinearFormat::Fp4E2M1PackedWithE8M0Scale {
                out_features: 1,
                in_features: 32,
                block_size: 32,
            }
        );
        let mut input = vec![0.0f32; 32];
        input[0] = 3.0;
        input[1] = 5.0;
        assert_eq!(linear.reference_matvec(&input).unwrap(), vec![13.0]);
    }

    #[test]
    fn fp8_source_linear_requires_scale_tensor() {
        let err = SourceLinearPayload::from_weight_and_scale(
            TensorRole::AttentionQuery,
            payload("fp8.weight", SourceDType::F8E4M3, vec![1, 1], vec![0x38]),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("requires E8M0 scale"));
    }

    fn payload(
        name: &str,
        dtype: SourceDType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> SourceTensorPayload {
        SourceTensorPayload {
            slice: SourceTensorSlice {
                name: name.into(),
                role: TensorRole::Unknown,
                path: PathBuf::from("synthetic.safetensors"),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype,
                shape,
            },
            bytes,
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}
