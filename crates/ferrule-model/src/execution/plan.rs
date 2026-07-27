/// Immutable, generation-stamped model preparation result.
///
/// Model-family implementations own their concrete prepared resources in `R`.
/// Runtime execution capabilities are published only after the runner is bound to
/// its backend and physical KV resources, so this preparation object deliberately
/// carries no second capability description.
#[derive(Debug)]
pub struct PreparedModel<R> {
    generation: u64,
    resources: R,
}

impl<R> PreparedModel<R> {
    pub const fn new(generation: u64, resources: R) -> Self {
        Self {
            generation,
            resources,
        }
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn resources(&self) -> &R {
        &self.resources
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_model_publishes_generation_and_immutable_resources() {
        let plan = PreparedModel::new(7, vec!["bound"]);

        assert_eq!(plan.generation(), 7);
        assert_eq!(plan.resources(), &["bound"]);
    }
}
