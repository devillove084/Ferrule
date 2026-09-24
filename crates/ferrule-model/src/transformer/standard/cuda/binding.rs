//! Image-scoped resident F32 parameters. No process-global or pointer-keyed cache.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::transformer::StandardTensorPlan;
use ferrule_backend::cuda::operators::linear::{
    CudaArtifactLinearHandle, CudaF32Buffer, CudaOperators,
};
use ferrule_common::{ParallelRankId, Result};

use crate::checkpoint::CheckpointDType;
use crate::nn::ParameterId;
use crate::transformer::{
    BoundParameter, PreparedDecoderGeneration, PreparedLinear, PreparedNorm, PreparedParameter,
    PreparedRope,
};

use super::cuda_error;

pub(super) struct ResidentLinear {
    pub handle: CudaArtifactLinearHandle,
    pub bias: Option<CudaF32Buffer>,
    pub shape: (usize, usize),
    bias_values: Option<Vec<f32>>,
}

pub(super) struct ResidentRope {
    pub table: PreparedRope,
    pub cosine: CudaF32Buffer,
    pub sine: CudaF32Buffer,
}

pub(super) struct Bindings {
    pub generation: PreparedDecoderGeneration,
    allowed: BTreeMap<ParameterId, BoundParameter>,
    linear: BTreeMap<(ParameterId, Option<crate::support::TensorRole>), Rc<ResidentLinear>>,
    vectors: BTreeMap<ParameterId, Rc<CudaF32Buffer>>,
    ropes: Vec<Rc<ResidentRope>>,
    bytes: usize,
    pub tensor: Option<(StandardTensorPlan, ParallelRankId)>,
}

impl Bindings {
    pub fn new(parameters: &[BoundParameter]) -> Result<Self> {
        let mut allowed = BTreeMap::<ParameterId, BoundParameter>::new();
        for parameter in parameters {
            if let Some(previous) = allowed.get(&parameter.canonical_id()) {
                if !previous.shares_storage_with(parameter) {
                    return Err(cuda_error("conflicting parameter image identities"));
                }
            } else {
                allowed.insert(parameter.canonical_id(), parameter.clone());
            }
        }
        Ok(Self {
            generation: PreparedDecoderGeneration::take()?,
            allowed,
            linear: BTreeMap::new(),
            vectors: BTreeMap::new(),
            ropes: Vec::new(),
            bytes: 0,
            tensor: None,
        })
    }

    fn validate(&self, parameter: &PreparedParameter) -> Result<()> {
        let Some(binding) = self.allowed.get(&parameter.canonical_id()) else {
            return Err(cuda_error(format!(
                "parameter '{}' is outside the assigned image/layers",
                parameter.binding().path()
            )));
        };
        if !binding.shares_storage_with(parameter.binding()) {
            return Err(cuda_error("parameter belongs to another prepared image"));
        }
        if !matches!(
            parameter.binding().weight().slice().dtype,
            CheckpointDType::F32 | CheckpointDType::Bf16
        ) || parameter.scale().is_some()
        {
            return Err(cuda_error(
                "standard CUDA supports only unscaled F32/BF16 weights",
            ));
        }
        Ok(())
    }

    pub fn linear(
        &mut self,
        ops: &CudaOperators,
        linear: &PreparedLinear,
    ) -> Result<Rc<ResidentLinear>> {
        self.validate(linear.parameter())?;
        match (&self.tensor, linear.tensor_shard()) {
            (Some((tensor, rank)), Some(shard)) => {
                if shard.plan() != &tensor.linear_plan(linear)? || shard.rank() != *rank {
                    return Err(cuda_error("TP weight plan/rank mismatch"));
                }
            }
            (None, None) => {}
            _ => {
                return Err(cuda_error(
                    "TP requires checkpoint-local weights; no upload-time slicing",
                ));
            }
        }
        if linear.tensor_shard().is_none()
            && linear.weight()?.execution.activation_quantization.is_some()
        {
            return Err(cuda_error(
                "F32 standard CUDA does not support activation quantization",
            ));
        }
        // Aliased matrices may have different TP layouts (e.g. Q versus O).
        // Keep TP caches role-qualified, while preserving TP1 alias deduplication.
        let id = (
            linear.parameter().canonical_id(),
            self.tensor.as_ref().map(|_| linear.role().clone()),
        );
        if let Some(resident) = self.linear.get(&id) {
            if resident.bias_values.as_deref() != linear.bias() {
                return Err(cuda_error("prepared linear bias changed within one image"));
            }
            return Ok(Rc::clone(resident));
        }
        // Representation conversion only; no CPU activation or matrix math.
        let (values, shape) = if let Some(shard) = linear.tensor_shard() {
            shard.validate_source_identity()?;
            (
                shard.values_f32()?,
                (shard.local_shape()[0], shard.local_shape()[1]),
            )
        } else {
            (
                linear.parameter().values_f32()?,
                (linear.out_features(), linear.in_features()),
            )
        };
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let handle = ops.upload_f32_linear(&bytes, shape.0, shape.1)?;
        let bias = linear
            .bias()
            .map(|v| ops.upload_f32_buffer(v))
            .transpose()?;
        self.bytes += bytes.len() + linear.bias().map_or(0, |v| v.len() * 4);
        let resident = Rc::new(ResidentLinear {
            handle,
            bias,
            shape,
            bias_values: linear.bias().map(<[f32]>::to_vec),
        });
        self.linear.insert(id, Rc::clone(&resident));
        Ok(resident)
    }

    pub fn vector(
        &mut self,
        ops: &CudaOperators,
        parameter: &PreparedParameter,
    ) -> Result<Rc<CudaF32Buffer>> {
        self.validate(parameter)?;
        let id = parameter.canonical_id();
        if let Some(resident) = self.vectors.get(&id) {
            return Ok(Rc::clone(resident));
        }
        let values = parameter.values_f32()?;
        let resident = Rc::new(ops.upload_f32_buffer(&values)?);
        self.bytes += values.len() * 4;
        self.vectors.insert(id, Rc::clone(&resident));
        Ok(resident)
    }

    pub fn norm(&mut self, ops: &CudaOperators, norm: &PreparedNorm) -> Result<Rc<CudaF32Buffer>> {
        self.vector(ops, norm.parameter())
    }

    pub fn rope(&mut self, ops: &CudaOperators, table: &PreparedRope) -> Result<Rc<ResidentRope>> {
        if let Some(resident) = self.ropes.iter().find(|r| r.table == *table) {
            return Ok(Rc::clone(resident));
        }
        let resident = Rc::new(ResidentRope {
            table: table.clone(),
            cosine: ops.upload_f32_buffer(table.cosine())?,
            sine: ops.upload_f32_buffer(table.sine())?,
        });
        self.bytes += (table.cosine().len() + table.sine().len()) * 4;
        self.ropes.push(Rc::clone(&resident));
        Ok(resident)
    }

    pub fn retain_on_unknown_completion(&mut self) {
        std::mem::forget(std::mem::take(&mut self.linear));
        std::mem::forget(std::mem::take(&mut self.vectors));
        std::mem::forget(std::mem::take(&mut self.ropes));
    }

    pub fn resident_bytes(&self) -> usize {
        self.bytes
    }
}
