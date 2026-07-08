//! Program-like generation API — builder for constrained generation.
//!
//! A `GenerationProgram` chains text segments, unconstrained generation,
//! and constrained generation into a single execution plan.

use crate::generation::{GenerationConfig, InferenceEngine};
use crate::sampling::constraint::TokenConstraint;
use crate::sampling::sampler::SamplingConfig;
use ferrule_model::runner::ModelRunner;

/// A generation program: chain text and constrained segments.
pub struct GenerationProgram {
    steps: Vec<ProgramStep>,
}

enum ProgramStep {
    /// Generate up to N tokens with given sampling config.
    Generate {
        max_tokens: usize,
        sampling: SamplingConfig,
    },
    /// Generate with a token constraint (e.g., JSON schema).
    Constrained {
        max_tokens: usize,
        _constraint: Box<dyn TokenConstraint>,
    },
    /// Append literal text (no generation).
    Literal(String),
}

impl GenerationProgram {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add an unconstrained generation step with default sampling.
    pub fn generate(mut self, max_tokens: usize) -> Self {
        self.steps.push(ProgramStep::Generate {
            max_tokens,
            sampling: SamplingConfig::default(),
        });
        self
    }

    /// Add an unconstrained generation step with custom sampling config.
    pub fn generate_with(mut self, max_tokens: usize, sampling: SamplingConfig) -> Self {
        self.steps.push(ProgramStep::Generate {
            max_tokens,
            sampling,
        });
        self
    }

    /// Add a constrained generation step.
    pub fn constrained(mut self, max_tokens: usize, constraint: Box<dyn TokenConstraint>) -> Self {
        self.steps.push(ProgramStep::Constrained {
            max_tokens,
            _constraint: constraint,
        });
        self
    }

    /// Append literal text (no generation).
    pub fn literal(mut self, text: impl Into<String>) -> Self {
        self.steps.push(ProgramStep::Literal(text.into()));
        self
    }

    /// Execute the program against a runner, collecting results.
    pub fn execute<R: ModelRunner>(
        &self,
        engine: &mut InferenceEngine<R>,
    ) -> anyhow::Result<Vec<String>> {
        let mut results = Vec::new();
        for step in &self.steps {
            match step {
                ProgramStep::Literal(text) => {
                    results.push(text.clone());
                }
                ProgramStep::Generate {
                    max_tokens,
                    sampling,
                } => {
                    let cfg = GenerationConfig {
                        max_new_tokens: *max_tokens,
                        ..Default::default()
                    };
                    // Temporarily swap the sampler config
                    let old_sampling = engine.sampler().config().clone();
                    engine.sampler_mut().config_mut().clone_from(sampling);
                    let result = engine.generate_text("", &cfg, |_| Ok(()))?;
                    engine.sampler_mut().config_mut().clone_from(&old_sampling);
                    results.push(result.text);
                }
                ProgramStep::Constrained { max_tokens, .. } => {
                    // Constrained generation — placeholder that records the step
                    results.push(format!("[constrained:{max_tokens}]"));
                }
            }
        }
        Ok(results)
    }

    /// Execute a program built for single-turn usage.
    /// Creates a temporary engine, runs all steps, and returns the collected output.
    pub fn execute_one_shot<R: ModelRunner + Default>(
        &self,
        runner: R,
        default_sampling: SamplingConfig,
    ) -> anyhow::Result<Vec<String>> {
        let mut engine = InferenceEngine::new(runner, default_sampling);
        self.execute(&mut engine)
    }

    /// Number of steps in the program.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the program has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl Default for GenerationProgram {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::constraint::{AllowListConstraint, StopStringConstraint};

    #[test]
    fn build_empty_program() {
        let prog = GenerationProgram::new();
        assert!(prog.is_empty());
        assert_eq!(prog.step_count(), 0);
    }

    #[test]
    fn build_program_with_steps() {
        let prog = GenerationProgram::new()
            .literal("Hello, ")
            .generate(10)
            .literal("!")
            .constrained(5, Box::new(AllowListConstraint::new(vec![1, 2, 3])))
            .generate(3);

        assert!(!prog.is_empty());
        assert_eq!(prog.step_count(), 5);
    }

    #[test]
    fn build_program_literal_only() {
        let prog = GenerationProgram::new()
            .literal("prefix-")
            .literal("middle-")
            .literal("suffix");

        assert_eq!(prog.step_count(), 3);
    }

    #[test]
    fn build_program_with_custom_sampling() {
        let sampling = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            ..SamplingConfig::default()
        };
        let prog = GenerationProgram::new()
            .literal("Prompt: ")
            .generate_with(20, sampling);

        assert_eq!(prog.step_count(), 2);
    }

    #[test]
    fn build_program_with_stop_constraint() {
        let prog = GenerationProgram::new()
            .literal("{")
            .constrained(50, Box::new(StopStringConstraint::new("</s>")))
            .literal("}");

        assert_eq!(prog.step_count(), 3);
    }

    #[test]
    fn program_default_is_empty() {
        let prog = GenerationProgram::default();
        assert!(prog.is_empty());
    }
}
