pub trait TokenConstraint {
    fn allow(&mut self, token: u32, token_text: &str) -> bool;
    fn reset(&mut self) {}
    fn is_exhausted(&self) -> bool {
        false
    }
}

pub struct SamplerMask {
    constraints: Vec<Box<dyn TokenConstraint + Send>>,
}

impl SamplerMask {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    pub fn add(&mut self, constraint: Box<dyn TokenConstraint + Send>) {
        self.constraints.push(constraint);
    }

    /// Returns allowed token ids from logits by applying all constraints.
    /// Unallowed tokens get -inf in the mask.
    pub fn apply(&mut self, logits: &[f32], token_texts: &[String]) -> Vec<f32> {
        let mut masked = logits.to_vec();
        for (token_id, (logit, text)) in masked.iter_mut().zip(token_texts.iter()).enumerate() {
            if !self
                .constraints
                .iter_mut()
                .all(|c| c.allow(token_id as u32, text))
            {
                *logit = f32::NEG_INFINITY;
            }
        }
        masked
    }

    pub fn reset(&mut self) {
        for c in &mut self.constraints {
            c.reset();
        }
    }
}

// Concrete constraints

pub struct MaxLengthConstraint {
    pub max_len: usize,
    pub current: usize,
}
impl TokenConstraint for MaxLengthConstraint {
    fn allow(&mut self, _token: u32, _text: &str) -> bool {
        if self.current >= self.max_len {
            return false;
        }
        self.current += 1;
        true
    }
    fn reset(&mut self) {
        self.current = 0;
    }
    fn is_exhausted(&self) -> bool {
        self.current >= self.max_len
    }
}

/// Simple JSON token-level constraint — only allows structural JSON tokens.
/// This is a placeholder; a full FSM-based JSON grammar would be needed for production.
pub struct JsonConstraint {
    depth: usize,
    in_string: bool,
    expect_key: bool,
    expect_value: bool,
}
impl JsonConstraint {
    pub fn new() -> Self {
        Self {
            depth: 0,
            in_string: false,
            expect_key: false,
            expect_value: false,
        }
    }
}
impl TokenConstraint for JsonConstraint {
    fn allow(&mut self, _token: u32, text: &str) -> bool {
        let t = text.trim();
        if t.contains('"') {
            self.in_string = !self.in_string;
            return true;
        }
        if self.in_string {
            return true;
        }
        if t.contains('{') {
            self.depth += 1;
            self.expect_key = true;
            return true;
        }
        if t.contains('}') {
            if self.depth == 0 {
                return false;
            }
            self.depth -= 1;
            return true;
        }
        if t.contains(':') {
            self.expect_value = true;
            self.expect_key = false;
            return true;
        }
        if t.contains(',') {
            self.expect_key = true;
            return true;
        }
        true // allow other tokens
    }
    fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn max_length_allows_and_blocks() {
        let mut c = MaxLengthConstraint {
            max_len: 3,
            current: 0,
        };
        assert!(c.allow(0, "a"));
        assert!(c.allow(1, "b"));
        assert!(c.allow(2, "c"));
        assert!(!c.allow(3, "d"));
    }
    #[test]
    fn sampler_mask_applies_constraints() {
        let mut mask = SamplerMask::new();
        mask.add(Box::new(MaxLengthConstraint {
            max_len: 2,
            current: 0,
        }));
        let logits = vec![1.0, 2.0, 3.0];
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let masked = mask.apply(&logits, &texts);
        assert!(masked[0] > f32::NEG_INFINITY);
        assert!(masked[1] > f32::NEG_INFINITY);
        assert!(masked[2] == f32::NEG_INFINITY);
    }
    #[test]
    fn json_allows_braces() {
        let mut c = JsonConstraint::new();
        assert!(c.allow(0, "{"));
        assert!(c.allow(1, "\"key\""));
        assert!(c.allow(2, ":"));
        assert!(c.allow(3, "}"));
    }

    #[test]
    fn json_rejects_unmatched_closing_brace() {
        let mut c = JsonConstraint::new();
        assert!(!c.allow(0, "}"));
    }
}
