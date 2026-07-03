//! Runtime-owned graph program bundle.
//!
//! A graph program keeps Ferrule's device-independent graph together with the
//! runtime semantic binding plan that resolves graph externals. Backend-specific
//! compiled artifacts should hang off future cached program objects rather than
//! being embedded in graph nodes.

use ferrule_core::Result;
use ferrule_graph::ComputeGraph;

use crate::graph_runtime::ExternalBindingPlan;
use crate::transformer_plan::TransformerRuntimePlan;

#[derive(Debug, Clone, PartialEq)]
pub struct GraphProgram {
    pub graph: ComputeGraph,
    pub bindings: ExternalBindingPlan,
    pub runtime_plan: TransformerRuntimePlan,
    pub profile: GraphProgramProfile,
}

impl GraphProgram {
    pub fn new(
        graph: ComputeGraph,
        bindings: ExternalBindingPlan,
        runtime_plan: TransformerRuntimePlan,
        profile: GraphProgramProfile,
    ) -> Result<Self> {
        graph.validate()?;
        Ok(Self {
            graph,
            bindings,
            runtime_plan,
            profile,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphProgramProfile {
    pub token_dim_symbol: String,
    pub max_batch_rows: Option<usize>,
    pub max_sequence_len: Option<usize>,
}

impl Default for GraphProgramProfile {
    fn default() -> Self {
        Self {
            token_dim_symbol: "tokens".into(),
            max_batch_rows: None,
            max_sequence_len: None,
        }
    }
}
