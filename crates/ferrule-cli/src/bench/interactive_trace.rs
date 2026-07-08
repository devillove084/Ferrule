//! Golden interactive traces for multi-turn chat correctness.
//!
//! A trace records expected token IDs and stop behaviour for a fixed set of
//! chat turns under greedy decoding. It serves as a regression gate: running
//! `bench-interactive --golden <trace>` compares the live run against the
//! golden file and reports mismatches.
//!
//! The trace format is intentionally simple (no extra deps beyond serde) so
//! it can be checked into version control and diffed easily.

use serde::{Deserialize, Serialize};

/// A single turn in a golden interactive trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenTurn {
    /// The raw user text (before chat-template wrapping).
    pub prompt_text: String,
    /// Token IDs after chat-template encoding. Empty if only comparing
    /// generated output (not tokenization).
    pub prompt_tokens: Vec<u32>,
    /// Expected generated token IDs for this turn.
    pub generated_tokens: Vec<u32>,
    /// Whether the turn was expected to stop because of EOS.
    pub stopped_by_eos: bool,
    /// Expected stop string that ended generation, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_by_string: Option<String>,
}

/// A golden interactive trace for a specific model + chat-template +
/// generation-config combination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractiveTrace {
    /// Human-readable label for this trace.
    pub label: String,
    /// Model directory or identifier used to produce the trace.
    pub model: String,
    /// Chat template name (e.g. "deepseek-v4").
    pub chat_template: String,
    /// Maximum new tokens per turn during trace recording.
    pub max_new_tokens: usize,
    /// The recorded turns.
    pub turns: Vec<GoldenTurn>,
}

/// Result of comparing a live interactive run against a golden trace.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractiveTraceComparison {
    pub label: String,
    pub turns_compared: usize,
    pub turns_ok: usize,
    pub mismatches: Vec<InteractiveTurnMismatch>,
}

impl InteractiveTraceComparison {
    pub fn all_ok(&self) -> bool {
        self.mismatches.is_empty() && self.turns_compared == self.turns_ok
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractiveTurnMismatch {
    pub turn_index: usize,
    pub prompt_text: String,
    pub message: String,
    pub expected_tokens: Vec<u32>,
    pub observed_tokens: Vec<u32>,
    pub expected_stopped_by_eos: bool,
    pub observed_stopped_by_eos: bool,
}

/// Compare a live interactive run result against a golden trace.
pub fn compare_interactive_trace(
    golden: &InteractiveTrace,
    observed_turns: &[GoldenTurn],
) -> InteractiveTraceComparison {
    let mut comparison = InteractiveTraceComparison {
        label: golden.label.clone(),
        turns_compared: golden.turns.len().min(observed_turns.len()),
        ..Default::default()
    };

    for i in 0..comparison.turns_compared {
        let expected = &golden.turns[i];
        let observed = &observed_turns[i];
        let mut ok = true;
        let mut messages: Vec<String> = Vec::new();

        if expected.generated_tokens != observed.generated_tokens {
            ok = false;
            messages.push(format!(
                "token mismatch: expected {:?}, observed {:?}",
                expected.generated_tokens, observed.generated_tokens
            ));
        }
        if expected.stopped_by_eos != observed.stopped_by_eos {
            ok = false;
            messages.push(format!(
                "stop mismatch: expected_eos={}, observed_eos={}",
                expected.stopped_by_eos, observed.stopped_by_eos
            ));
        }
        if expected.stopped_by_string != observed.stopped_by_string {
            ok = false;
            messages.push(format!(
                "stop-string mismatch: expected={:?}, observed={:?}",
                expected.stopped_by_string, observed.stopped_by_string
            ));
        }

        if ok {
            comparison.turns_ok += 1;
        } else {
            comparison.mismatches.push(InteractiveTurnMismatch {
                turn_index: i,
                prompt_text: expected.prompt_text.clone(),
                message: messages.join("; "),
                expected_tokens: expected.generated_tokens.clone(),
                observed_tokens: observed.generated_tokens.clone(),
                expected_stopped_by_eos: expected.stopped_by_eos,
                observed_stopped_by_eos: observed.stopped_by_eos,
            });
        }
    }

    comparison
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_turn(text: &str, tokens: Vec<u32>, eos: bool) -> GoldenTurn {
        GoldenTurn {
            prompt_text: text.into(),
            prompt_tokens: Vec::new(),
            generated_tokens: tokens,
            stopped_by_eos: eos,
            stopped_by_string: None,
        }
    }

    #[test]
    fn comparison_all_match() {
        let golden = InteractiveTrace {
            label: "test".into(),
            model: "model".into(),
            chat_template: "plain".into(),
            max_new_tokens: 1,
            turns: vec![make_turn("Hello", vec![42], false)],
        };
        let observed = vec![make_turn("Hello", vec![42], false)];
        let cmp = compare_interactive_trace(&golden, &observed);
        assert!(cmp.all_ok());
        assert_eq!(cmp.turns_compared, 1);
        assert_eq!(cmp.turns_ok, 1);
    }

    #[test]
    fn comparison_token_mismatch() {
        let golden = InteractiveTrace {
            label: "test".into(),
            model: "model".into(),
            chat_template: "plain".into(),
            max_new_tokens: 1,
            turns: vec![make_turn("Hello", vec![42], false)],
        };
        let observed = vec![make_turn("Hello", vec![99], false)];
        let cmp = compare_interactive_trace(&golden, &observed);
        assert!(!cmp.all_ok());
        assert_eq!(cmp.mismatches.len(), 1);
    }

    #[test]
    fn comparison_stop_mismatch() {
        let golden = InteractiveTrace {
            label: "test".into(),
            model: "model".into(),
            chat_template: "plain".into(),
            max_new_tokens: 1,
            turns: vec![make_turn("Hello", vec![42], true)],
        };
        let observed = vec![make_turn("Hello", vec![42], false)];
        let cmp = compare_interactive_trace(&golden, &observed);
        assert!(!cmp.all_ok());
        assert_eq!(cmp.mismatches.len(), 1);
    }
}
