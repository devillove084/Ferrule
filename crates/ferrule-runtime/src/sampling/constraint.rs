//! Token constraint / mask API — foundation for structured decoding.
//!
//! Constraints can reject individual tokens before sampling.
//! Composable: chain multiple constraints with AND logic.

/// A token-level constraint: can this token be sampled at this point?
pub trait TokenConstraint: Send {
    /// Return true if `token` is allowed. `decoded_text` is the cumulative
    /// decoded output so far (used for stop-string / grammar matching).
    fn allow(&mut self, token: u32, decoded_text: &str) -> bool;

    /// Optional: update internal state after a token is accepted.
    fn accept(&mut self, _token: u32, _decoded_text: &str) {}
}

/// Composite constraint: all sub-constraints must allow the token.
pub struct AllConstraint {
    children: Vec<Box<dyn TokenConstraint>>,
}

impl Default for AllConstraint {
    fn default() -> Self {
        Self::new()
    }
}

impl AllConstraint {
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    pub fn push(&mut self, c: Box<dyn TokenConstraint>) {
        self.children.push(c);
    }
}

impl TokenConstraint for AllConstraint {
    fn allow(&mut self, token: u32, text: &str) -> bool {
        self.children.iter_mut().all(|c| c.allow(token, text))
    }

    fn accept(&mut self, token: u32, text: &str) {
        for c in &mut self.children {
            c.accept(token, text);
        }
    }
}

/// Stop when a specific string appears in decoded output.
pub struct StopStringConstraint {
    stop: String,
    found: bool,
}

impl StopStringConstraint {
    pub fn new(stop: impl Into<String>) -> Self {
        Self {
            stop: stop.into(),
            found: false,
        }
    }
}

impl TokenConstraint for StopStringConstraint {
    fn allow(&mut self, _token: u32, text: &str) -> bool {
        if text.contains(&self.stop) {
            self.found = true;
        }
        !self.found
    }
}

/// Only allow tokens within a fixed allowlist (e.g., for constrained vocab).
pub struct AllowListConstraint {
    allowed: Vec<u32>,
}

impl AllowListConstraint {
    pub fn new(allowed: Vec<u32>) -> Self {
        Self { allowed }
    }
}

impl TokenConstraint for AllowListConstraint {
    fn allow(&mut self, token: u32, _text: &str) -> bool {
        self.allowed.contains(&token)
    }
}

/// Never-allow constraint (for testing).
pub struct DenyAll;

impl TokenConstraint for DenyAll {
    fn allow(&mut self, _token: u32, _text: &str) -> bool {
        false
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_string_blocks() {
        let mut c = StopStringConstraint::new("</s>");
        assert!(c.allow(1, "hello"));
        assert!(c.allow(2, "hello wor"));
        assert!(!c.allow(3, "hello world</s>"));
    }

    #[test]
    fn allowlist_filters() {
        let mut c = AllowListConstraint::new(vec![42, 99]);
        assert!((&mut c as &mut dyn TokenConstraint).allow(42, ""));
        assert!((&mut c as &mut dyn TokenConstraint).allow(99, ""));
        assert!(!(&mut c as &mut dyn TokenConstraint).allow(1, ""));
    }

    #[test]
    fn all_composite() {
        let mut all = AllConstraint::new();
        all.push(Box::new(AllowListConstraint::new(vec![1, 2, 3])));
        all.push(Box::new(StopStringConstraint::new("stop")));

        assert!(all.allow(1, "go"));
        assert!(all.allow(2, "go"));
        assert!(!all.allow(4, "go")); // not in allowlist
        assert!(!all.allow(1, "stop")); // stop string found
    }

    #[test]
    fn deny_all() {
        let mut d = DenyAll;
        assert!(!(&mut d as &mut dyn TokenConstraint).allow(0, ""));
        assert!(!(&mut d as &mut dyn TokenConstraint).allow(999, ""));
    }

    #[test]
    fn composite_accept_propagates() {
        let mut all = AllConstraint::new();
        all.push(Box::new(AllowListConstraint::new(vec![1])));
        all.accept(1, "hi");
        // No panic — accept should propagate
    }
}
