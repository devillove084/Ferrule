use std::sync::Arc;

use ferrule_common::Result;

use crate::support::TensorRole;

use super::{
    BoundParameter, PreparedHybridRankParameter, PreparedLinear, PreparedParameter,
    PreparedParameterStorage, StateDictMaterializer, model_error,
};

impl PreparedParameter {
    pub fn hybrid_rank(&self) -> Option<&PreparedHybridRankParameter> {
        match &self.weight {
            PreparedParameterStorage::HybridRank(value) => Some(value),
            _ => None,
        }
    }
}

impl PreparedLinear {
    /// Build the replacement without changing any existing Arc alias. The owner
    /// retains the original image until every replacement has been constructed.
    pub(crate) fn mixed_runtime_binding(
        &self,
        completion: &crate::decoder::MixedSourceRetirement,
    ) -> Result<Self> {
        if !completion.is_complete() {
            return Err(model_error(
                "Mixed host source retirement before final claim completion",
            ));
        }
        let Some(rank) = self.hybrid_rank() else {
            return Ok(self.clone());
        };
        let super::HybridRankStorage::Numeric(payload) = rank.storage() else {
            return Ok(self.clone());
        };
        if !rank.layout().precision().is_mixed() || self.bias.is_some() {
            return Err(model_error(
                "host source retirement requires an unbiased Mixed rank projection",
            ));
        }
        payload
            .plan()
            .validate_binding(self.parameter.binding(), rank.layout())?;
        let mut metadata = rank.clone();
        metadata.storage = super::HybridRankStorage::NumericBinding(payload.plan().clone());
        let parameter = Arc::new(PreparedParameter {
            binding: self.parameter.binding().clone(),
            weight: PreparedParameterStorage::HybridRank(Arc::new(metadata)),
            scale: None,
        });
        Self::from_parameter(parameter, self.role.clone())
    }

    pub fn hybrid_rank(&self) -> Option<&PreparedHybridRankParameter> {
        self.parameter.hybrid_rank()
    }
}

impl StateDictMaterializer {
    pub(super) fn prepare_cuda_rank_linear(
        &self,
        binding: &BoundParameter,
        role: TensorRole,
    ) -> Result<PreparedLinear> {
        PreparedLinear::from_parameter(self.cuda_rank_parameter(binding)?, role)
    }

    // A profile/role miss may fall through; a selected rank read failure may not.
    pub(super) fn cuda_rank_layer_parameter(
        &self,
        binding: &BoundParameter,
    ) -> Option<Result<Arc<PreparedParameter>>> {
        if self.cuda_rank.is_some()
            && matches!(
                binding.role(),
                TensorRole::LinearAttentionConv
                    | TensorRole::LinearAttentionALog
                    | TensorRole::LinearAttentionDtBias
            )
        {
            Some(self.cuda_rank_parameter(binding))
        } else {
            None
        }
    }

    pub(super) fn validate_cuda_rank_full_read(&self, binding: &BoundParameter) -> Result<()> {
        if self.cuda_rank.is_some()
            && !matches!(
                binding.role(),
                TensorRole::TokenEmbedding
                    | TensorRole::OutputNorm
                    | TensorRole::AttentionNorm
                    | TensorRole::FeedForwardNorm
                    | TensorRole::AttentionQueryNorm
                    | TensorRole::AttentionKeyNorm
                    | TensorRole::LinearAttentionNorm
            )
        {
            return Err(model_error(
                "CUDA hybrid rank prohibits full projection/expert reads",
            ));
        }
        Ok(())
    }
}

/// Backend-compiled linear handle with its activation contract.
pub struct PreparedCudaLinear {
    pub(crate) handle: ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle,
    pub(crate) activation_quantization: Option<crate::checkpoint::weight::ActivationQuantization>,
}

impl PreparedCudaLinear {
    pub fn new(
        handle: ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle,
        activation_quantization: Option<crate::checkpoint::weight::ActivationQuantization>,
    ) -> Self {
        Self {
            handle,
            activation_quantization,
        }
    }

    pub const fn handle(
        &self,
    ) -> &ferrule_backend::cuda::operators::linear::CudaArtifactLinearHandle {
        &self.handle
    }

    pub const fn activation_quantization(
        &self,
    ) -> Option<crate::checkpoint::weight::ActivationQuantization> {
        self.activation_quantization
    }
}
