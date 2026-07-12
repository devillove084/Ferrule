use ferrule_common::{Error, Result};

/// Concrete device backend used to execute model operators.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelExecutionBackend {
    /// CPU/reference execution for every operator.
    #[default]
    Cpu,
    /// CUDA execution for model operators.
    Cuda,
}

impl ModelExecutionBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu-reference",
            Self::Cuda => "cuda",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "cpu" | "cpu-reference" | "reference" => Ok(Self::Cpu),
            "cuda" | "cuda-hybrid" | "gpu" => Ok(Self::Cuda),
            other => Err(Error::Model(format!(
                "unknown model execution backend '{other}' (expected cpu or cuda)"
            ))),
        }
    }
}
