use ferrule_core::{FerruleError, FerruleResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub initial_observation: String,
    pub expected_final: Option<String>,
}

pub fn load_agent_tasks_jsonl<P: AsRef<Path>>(path: P) -> FerruleResult<Vec<AgentTask>> {
    let raw = fs::read_to_string(path)?;
    let mut out = Vec::new();

    for (idx, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let task = serde_json::from_str::<AgentTask>(line).map_err(|e| {
            FerruleError::Config(format!("invalid agent jsonl at line {}: {e}", idx + 1))
        })?;
        out.push(task);
    }

    if out.is_empty() {
        return Err(FerruleError::Config("agent dataset is empty".to_string()));
    }

    Ok(out)
}
