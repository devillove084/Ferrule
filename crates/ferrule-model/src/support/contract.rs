use crate::spec::TransformerSpec;
use crate::tensor_policy::TensorClassCount;

use super::binding::TensorBinding;
use super::layout::ModelLayout;
use super::plan::EnginePlan;
use super::policies::{PolicySet, SpeculationMode};
use super::roles::TensorRole;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSupportContract {
    pub spec: TransformerSpec,
    pub layout: ModelLayout,
    pub tensor_bindings: Vec<TensorBinding>,
    pub policies: PolicySet,
}

impl ModelSupportContract {
    pub fn from_spec(spec: &TransformerSpec, tensor_classes: &[TensorClassCount]) -> Self {
        let layout = ModelLayout::from_spec(spec);
        let tensor_bindings = tensor_classes
            .iter()
            .map(TensorBinding::from_class_count)
            .collect();
        let mut policies = PolicySet::from_spec(spec);
        if tensor_classes
            .iter()
            .any(|item| item.class.is_speculative())
        {
            policies.speculation.mode = SpeculationMode::MultiTokenPrediction;
        }
        Self {
            spec: spec.clone(),
            layout,
            tensor_bindings,
            policies,
        }
    }

    pub fn with_speculation_mode(mut self, mode: SpeculationMode) -> Self {
        self.policies.speculation.mode = mode;
        self
    }

    pub fn engine_plan(&self) -> EnginePlan {
        EnginePlan::from_contract(self)
    }

    pub fn bound_roles(&self) -> Vec<TensorRole> {
        let mut roles = self
            .tensor_bindings
            .iter()
            .map(|binding| binding.role.clone())
            .collect::<Vec<_>>();
        roles.sort();
        roles.dedup();
        roles
    }
}
