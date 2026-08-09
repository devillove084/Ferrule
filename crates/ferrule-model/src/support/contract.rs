use crate::spec::TransformerSpec;
use crate::tensor_policy::TensorClassCount;

use super::binding::TensorBinding;
use super::layout::ModelLayout;
use super::plan::EnginePlan;
use super::policies::{PolicySet, SpeculationMode};
use super::roles::TensorRole;
use super::validation::{LayoutValidationReport, validate_model_layout_bindings};

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
            .map(|count| TensorBinding::from_class_count_for_family(count, &spec.family))
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

    pub fn with_role_alias(mut self, logical_role: TensorRole, physical_role: TensorRole) -> Self {
        self.layout.add_role_alias(logical_role, physical_role);
        self
    }

    pub fn validate_layout_bindings(&self) -> LayoutValidationReport {
        validate_model_layout_bindings(&self.layout, &self.tensor_bindings)
    }

    pub fn engine_plan(&self) -> EnginePlan {
        EnginePlan::from_contract(self)
    }

    pub fn bound_roles(&self) -> Vec<TensorRole> {
        self.validate_layout_bindings()
            .bound_role_counts
            .into_iter()
            .map(|count| count.role)
            .collect()
    }
}
