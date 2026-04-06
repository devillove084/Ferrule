use ferrule_core::{FerruleError, FerruleResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptExample {
    pub prompt: String,
    pub expected: Option<String>,
}

pub fn load_jsonl_dataset<P: AsRef<Path>>(path: P) -> FerruleResult<Vec<PromptExample>> {
    let raw = fs::read_to_string(path)?;
    let mut out = Vec::new();

    for (idx, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let item = serde_json::from_str::<PromptExample>(line)
            .map_err(|e| FerruleError::Config(format!("invalid jsonl at line {}: {e}", idx + 1)))?;

        out.push(item);
    }

    if out.is_empty() {
        return Err(FerruleError::Config("dataset is empty".to_string()));
    }

    Ok(out)
}
