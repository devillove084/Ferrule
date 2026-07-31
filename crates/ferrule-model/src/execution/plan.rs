use super::PreparedExecutable;

/// Immutable, generation-stamped model preparation result.
///
/// `R` contains model-family bindings used by host/reference execution. The
/// executable is the single scheduling contract: every backend consumes its ordered
/// stages and exact resource access declarations rather than inferring residency
/// from opaque model fields.
#[derive(Debug)]
pub struct PreparedModel<R, O> {
    generation: u64,
    resources: R,
    executable: PreparedExecutable<O>,
}

impl<R, O> PreparedModel<R, O> {
    pub const fn new(generation: u64, resources: R, executable: PreparedExecutable<O>) -> Self {
        Self {
            generation,
            resources,
            executable,
        }
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn resources(&self) -> &R {
        &self.resources
    }

    pub const fn executable(&self) -> &PreparedExecutable<O> {
        &self.executable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{ExecutableStage, TransformerStage, WorkspaceClaim};

    #[test]
    fn prepared_model_publishes_generation_resources_and_single_executable() {
        let executable = PreparedExecutable::new(
            [],
            [ExecutableStage::new(
                TransformerStage::Attachment { index: 0 },
                [],
                WorkspaceClaim::NONE,
            )],
        )
        .unwrap();
        let plan = PreparedModel::new(7, vec!["bound"], executable);

        assert_eq!(plan.generation(), 7);
        assert_eq!(plan.resources(), &["bound"]);
        assert_eq!(plan.executable().stages().len(), 1);
    }
}
