//! Production numeric FP8 choices; storage is not an arithmetic policy.
use ferrule_model::transformer::NumericFp8Precision;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericFp8Profile {
    #[default]
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "w8a16-bf16")]
    W8A16Bf16,
}

impl NumericFp8Profile {
    pub const fn precision(self) -> NumericFp8Precision {
        match self {
            Self::F32 => NumericFp8Precision::F32Tf32x3,
            Self::W8A16Bf16 => NumericFp8Precision::W8A16Bf16F32Accumulate,
        }
    }

    pub const fn resident_backend_name(self) -> &'static str {
        match self {
            Self::F32 => "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-35b-a3b",
            Self::W8A16Bf16 => "cuda-hybrid-numeric-fp8-w8a16-bf16-f32-accumulate-qwen35-35b-a3b",
        }
    }

    pub const fn expert_backend_name(self, process: bool) -> &'static str {
        match (self, process) {
            (Self::F32, false) => "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-thread-ep",
            (Self::F32, true) => "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-process-ep",
            (Self::W8A16Bf16, false) => {
                "cuda-hybrid-numeric-fp8-w8a16-bf16-f32-accumulate-qwen35-thread-ep"
            }
            (Self::W8A16Bf16, true) => {
                "cuda-hybrid-numeric-fp8-w8a16-bf16-f32-accumulate-qwen35-process-ep"
            }
        }
    }

    pub const fn arithmetic_description(self) -> &'static str {
        match self {
            Self::F32 => {
                "F32Tf32x3; F32 operands; fused F32 M1 / TF32x3 multirow; F32 accumulation/output/model/KV/state boundaries"
            }
            Self::W8A16Bf16 => {
                "W8A16Bf16F32Accumulate; BF16-RNE operands; E4M3FN times original numeric scale rounded F32 then BF16; fused BF16 MMA; F32 accumulation/output/model/KV/state boundaries; required global numeric workspace=0; not end-to-end BF16 or same-precision Marlin"
            }
        }
    }
}

impl From<NumericFp8Profile> for NumericFp8Precision {
    fn from(profile: NumericFp8Profile) -> Self {
        profile.precision()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/numeric_profile.rs"]
mod tests;
