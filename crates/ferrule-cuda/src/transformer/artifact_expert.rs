//! Artifact-preserved packed FP4 expert execution primitives.
//!
//! This is the first CUDA execution surface for artifact-format packed routed experts.
//! It deliberately stays generic: it accepts device buffers plus explicit shapes,
//! not model-family tensor names. Scheduler/residency code can map a resident
//! expert handle to these buffers later.

use cuda_core::{stream::CudaStream, DeviceBuffer, LaunchConfig};
use ferrule_core::{Error, Result};

use crate::context::cu;
use crate::kernels::kernels::LoadedModule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPackedFp4LinearShape {
    pub out_features: usize,
    pub in_features: usize,
    pub packed_offset: usize,
    pub scale_offset: usize,
}

impl CudaPackedFp4LinearShape {
    pub fn new(out_features: usize, in_features: usize) -> Self {
        Self {
            out_features,
            in_features,
            packed_offset: 0,
            scale_offset: 0,
        }
    }

    pub fn with_offsets(mut self, packed_offset: usize, scale_offset: usize) -> Self {
        self.packed_offset = packed_offset;
        self.scale_offset = scale_offset;
        self
    }

    pub fn validate(&self, label: &str) -> Result<()> {
        if self.in_features == 0
            || !self.in_features.is_multiple_of(32)
            || !self.in_features.is_multiple_of(2)
        {
            return Err(Error::Internal(format!(
                "{label} artifact FP4 linear requires in_features divisible by 32, got {}",
                self.in_features
            )));
        }
        if self.out_features == 0 {
            return Err(Error::Internal(format!(
                "{label} artifact FP4 linear has zero out_features"
            )));
        }
        let packed_len = self
            .out_features
            .checked_mul(self.in_features / 2)
            .ok_or_else(|| Error::Internal(format!("{label} packed FP4 size overflow")))?;
        let scale_len = self
            .out_features
            .checked_mul(self.in_features / 32)
            .ok_or_else(|| Error::Internal(format!("{label} FP4 scale size overflow")))?;
        checked_u32(
            self.packed_offset.saturating_add(packed_len),
            label,
            "packed end",
        )?;
        checked_u32(
            self.scale_offset.saturating_add(scale_len),
            label,
            "scale end",
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct CudaPackedFp4Linear<'a> {
    pub packed: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub shape: CudaPackedFp4LinearShape,
}

impl<'a> CudaPackedFp4Linear<'a> {
    pub fn new(
        packed: &'a DeviceBuffer<u8>,
        scales: &'a DeviceBuffer<u8>,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        Self {
            packed,
            scales,
            shape: CudaPackedFp4LinearShape::new(out_features, in_features),
        }
    }

    pub fn with_offsets(mut self, packed_offset: usize, scale_offset: usize) -> Self {
        self.shape = self.shape.with_offsets(packed_offset, scale_offset);
        self
    }

    pub fn validate(&self, label: &str) -> Result<()> {
        self.shape.validate(label)
    }

    pub fn out_features(&self) -> usize {
        self.shape.out_features
    }

    pub fn in_features(&self) -> usize {
        self.shape.in_features
    }

    pub fn packed_offset(&self) -> usize {
        self.shape.packed_offset
    }

    pub fn scale_offset(&self) -> usize {
        self.shape.scale_offset
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPackedFp4ExpertShape {
    pub gate: CudaPackedFp4LinearShape,
    pub up: CudaPackedFp4LinearShape,
    pub down: CudaPackedFp4LinearShape,
}

impl CudaPackedFp4ExpertShape {
    pub fn validate(&self) -> Result<()> {
        self.gate.validate("gate")?;
        self.up.validate("up")?;
        self.down.validate("down")?;
        if self.gate.in_features != self.up.in_features {
            return Err(Error::Internal(format!(
                "packed FP4 expert gate/up input mismatch: {} vs {}",
                self.gate.in_features, self.up.in_features
            )));
        }
        if self.gate.out_features != self.up.out_features {
            return Err(Error::Internal(format!(
                "packed FP4 expert gate/up output mismatch: {} vs {}",
                self.gate.out_features, self.up.out_features
            )));
        }
        if self.down.in_features != self.gate.out_features {
            return Err(Error::Internal(format!(
                "packed FP4 expert down input {} must match gate/up output {}",
                self.down.in_features, self.gate.out_features
            )));
        }
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.gate.in_features
    }

    pub fn intermediate_size(&self) -> usize {
        self.gate.out_features
    }

    pub fn output_size(&self) -> usize {
        self.down.out_features
    }
}

#[derive(Clone, Copy)]
pub struct CudaPackedFp4Expert<'a> {
    pub gate: CudaPackedFp4Linear<'a>,
    pub up: CudaPackedFp4Linear<'a>,
    pub down: CudaPackedFp4Linear<'a>,
}

impl CudaPackedFp4Expert<'_> {
    pub fn shape(&self) -> CudaPackedFp4ExpertShape {
        CudaPackedFp4ExpertShape {
            gate: self.gate.shape,
            up: self.up.shape,
            down: self.down.shape,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.shape().validate()
    }

    pub fn hidden_size(&self) -> usize {
        self.shape().hidden_size()
    }

    pub fn intermediate_size(&self) -> usize {
        self.shape().intermediate_size()
    }

    pub fn output_size(&self) -> usize {
        self.shape().output_size()
    }
}

pub struct CudaPackedFp4ExpertScratch {
    pub gate: DeviceBuffer<f32>,
    pub up: DeviceBuffer<f32>,
    pub hidden: DeviceBuffer<f32>,
}

impl CudaPackedFp4ExpertScratch {
    pub fn new(stream: &CudaStream, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            gate: cu(DeviceBuffer::<f32>::zeroed(stream, intermediate_size))?,
            up: cu(DeviceBuffer::<f32>::zeroed(stream, intermediate_size))?,
            hidden: cu(DeviceBuffer::<f32>::zeroed(stream, intermediate_size))?,
        })
    }
}

#[derive(Clone, Copy)]
pub struct CudaPackedFp4ExpertExecutor<'a> {
    pub module: &'a LoadedModule,
    pub stream: &'a CudaStream,
    pub swiglu_limit: f32,
}

impl<'a> CudaPackedFp4ExpertExecutor<'a> {
    pub fn new(module: &'a LoadedModule, stream: &'a CudaStream, swiglu_limit: f32) -> Self {
        Self {
            module,
            stream,
            swiglu_limit,
        }
    }

    pub fn execute(
        &self,
        expert: &CudaPackedFp4Expert<'_>,
        input: &DeviceBuffer<f32>,
        route_weight: f32,
        scratch: &mut CudaPackedFp4ExpertScratch,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        expert.validate()?;
        let mid = expert.intermediate_size();
        let out = expert.output_size();
        let cfg_mid = LaunchConfig::for_num_elems(mid as u32);
        let cfg_out = LaunchConfig::for_num_elems(out as u32);

        cu(self.module.gemv_dual_fp4_e2m1_e8m0_off(
            self.stream,
            cfg_mid,
            input,
            expert.gate.packed,
            expert.gate.scales,
            &mut scratch.gate,
            checked_u32(expert.gate.packed_offset(), "gate", "packed offset")?,
            checked_u32(expert.gate.scale_offset(), "gate", "scale offset")?,
            expert.up.packed,
            expert.up.scales,
            &mut scratch.up,
            checked_u32(expert.up.packed_offset(), "up", "packed offset")?,
            checked_u32(expert.up.scale_offset(), "up", "scale offset")?,
            checked_u32(mid, "expert", "intermediate size")?,
            checked_u32(expert.hidden_size(), "expert", "hidden size")?,
        ))?;

        cu(self.module.swiglu_weighted_clamped(
            self.stream,
            cfg_mid,
            &scratch.gate,
            &scratch.up,
            &mut scratch.hidden,
            checked_u32(mid, "expert", "intermediate size")?,
            route_weight,
            self.swiglu_limit,
        ))?;

        cu(self.module.gemv_fp4_e2m1_e8m0_off(
            self.stream,
            cfg_out,
            &scratch.hidden,
            expert.down.packed,
            expert.down.scales,
            output,
            checked_u32(out, "down", "output size")?,
            checked_u32(expert.down.in_features(), "down", "input size")?,
            checked_u32(expert.down.packed_offset(), "down", "packed offset")?,
            checked_u32(expert.down.scale_offset(), "down", "scale offset")?,
        ))
    }
}

fn checked_u32(value: usize, label: &str, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::Internal(format!(
            "{label} artifact FP4 {field} exceeds CUDA u32 launch ABI: {value}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_fp4_expert_shape_validation_accepts_large_moe_shapes() {
        let shape = CudaPackedFp4ExpertShape {
            gate: CudaPackedFp4LinearShape::new(2048, 4096),
            up: CudaPackedFp4LinearShape::new(2048, 4096),
            down: CudaPackedFp4LinearShape::new(4096, 2048),
        };
        assert!(shape.validate().is_ok());
        assert_eq!(shape.hidden_size(), 4096);
        assert_eq!(shape.intermediate_size(), 2048);
        assert_eq!(shape.output_size(), 4096);
    }

    #[test]
    fn packed_fp4_expert_shape_validation_rejects_bad_down_input() {
        let shape = CudaPackedFp4ExpertShape {
            gate: CudaPackedFp4LinearShape::new(8, 32),
            up: CudaPackedFp4LinearShape::new(8, 32),
            down: CudaPackedFp4LinearShape::new(32, 32),
        };
        let err = shape.validate().unwrap_err();
        assert!(err.to_string().contains("down input"));
    }
}
