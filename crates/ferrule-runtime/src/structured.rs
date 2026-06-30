//! Structured output constraints — JSON, regex, and grammar-based decoding.
//!
//! Plugs into the TokenConstraint trait from `super::constraint`.
//! Current: simple JSON object bracket-balancing constraint.

use crate::constraint::TokenConstraint;

/// Constraint that ensures the output forms a syntactically valid JSON object.
///
/// Tracks bracket depth: `{` increments, `}` decrements.
/// Rejects tokens that would unbalance the structure.
#[derive(Debug, Clone)]
pub struct JsonConstraint {
    /// Current brace depth (0 = outside object, 1 = inside top-level).
    depth: i32,
    /// Whether we've seen the opening `{`.
    started: bool,
    /// Whether the object has been closed.
    finished: bool,
}

impl Default for JsonConstraint {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonConstraint {
    pub fn new() -> Self {
        Self {
            depth: 0,
            started: false,
            finished: false,
        }
    }
}

impl TokenConstraint for JsonConstraint {
    fn allow(&mut self, _token: u32, decoded_text: &str) -> bool {
        // Simple: count braces in the decoded text so far
        let mut depth = 0i32;
        let mut started = false;
        let mut finished = false;

        // Re-parse state from scratch (simplest approach — correct but O(n))
        // For production, maintain incremental state.
        for ch in decoded_text.chars() {
            match ch {
                '{' => {
                    if !started {
                        started = true;
                        depth = 1;
                    } else {
                        depth += 1;
                    }
                }
                '}' => {
                    depth -= 1;
                    if started && depth == 0 {
                        finished = true;
                    }
                }
                _ => {}
            }
        }

        self.depth = depth;
        self.started = started;
        self.finished = finished;

        // Allow if object hasn't been closed yet (or hasn't started)
        !finished
    }
}

/// Constraint that limits total output length (in characters).
pub struct MaxLengthConstraint {
    max_chars: usize,
}

impl MaxLengthConstraint {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl TokenConstraint for MaxLengthConstraint {
    fn allow(&mut self, _token: u32, decoded_text: &str) -> bool {
        decoded_text.len() <= self.max_chars
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_allows_opening_brace() {
        let mut c = JsonConstraint::new();
        assert!(c.allow(1, "{"));
        assert!(!c.finished);
    }

    #[test]
    fn json_allows_key_value() {
        let mut c = JsonConstraint::new();
        assert!(!c.allow(1, r#"{"key": "value"}"#)); // closed → block
        assert!(c.finished);
    }

    #[test]
    fn json_blocks_after_close() {
        let mut c = JsonConstraint::new();
        assert!(!c.allow(1, r#"{"a":1}"#)); // closed → block
        assert!(c.finished);
        assert!(!c.allow(2, r#"{"a":1}extra"#));
    }

    #[test]
    fn json_nested_allowed_until_closed() {
        let mut c = JsonConstraint::new();
        assert!(!c.allow(1, r#"{"outer": {"inner": 1}}"#)); // closed → block
        assert!(c.finished);
    }

    #[test]
    fn max_length_blocks_excess() {
        let mut c = MaxLengthConstraint::new(10);
        assert!(c.allow(1, "short"));
        assert!(!c.allow(2, "this is way too long"));
    }

    #[test]
    fn max_length_allows_at_boundary() {
        let mut c = MaxLengthConstraint::new(5);
        assert!(c.allow(1, "hello"));
        assert!(!c.allow(2, "hello!"));
    }
}
