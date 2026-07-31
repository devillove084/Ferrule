//! HF `config.json` field extraction helpers shared by family config parsers.

/// First present key coerced to `usize`, or `None`.
pub(crate) fn usize_key(json: &serde_json::Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
    })
}

/// First present key coerced to `f32`, or `None`.
pub(crate) fn f32_key(json: &serde_json::Value, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|key| {
        json.get(*key)
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
    })
}
