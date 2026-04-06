use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use ferrule_core::{FerruleError, FerruleResult};

pub struct PolicyBiasHead {
    varmap: VarMap,
    bias: Tensor,
    vocab_size: usize,
    device: Device,
}

impl PolicyBiasHead {
    pub fn new(vocab_size: usize, device: &Device) -> FerruleResult<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let bias = vb
            .get((vocab_size,), "policy_bias")
            .map_err(|e| FerruleError::Model(format!("failed to init policy bias: {e}")))?;

        Ok(Self {
            varmap,
            bias,
            vocab_size,
            device: device.clone(),
        })
    }

    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn trainable_vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    pub fn apply_to_logits(&self, logits: &Tensor) -> FerruleResult<Tensor> {
        let adjusted = if logits.rank() == 2 {
            let b = self
                .bias
                .unsqueeze(0)
                .map_err(|e| FerruleError::Model(format!("failed to unsqueeze bias: {e}")))?;

            logits
                .broadcast_add(&b)
                .map_err(|e| FerruleError::Model(format!("failed to add bias to logits: {e}")))?
        } else {
            logits
                .broadcast_add(&self.bias)
                .map_err(|e| FerruleError::Model(format!("failed to add bias to logits: {e}")))?
        };

        Ok(adjusted)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
