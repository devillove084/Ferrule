use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EpisodeStatus {
    Running,
    WaitingTool,
    Finished,
    Truncated,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeContext {
    pub step_idx: usize,
    pub max_steps: usize,
    pub status: EpisodeStatus,
    pub transcript: Vec<String>,
}

impl EpisodeContext {
    pub fn new(max_steps: usize) -> Self {
        Self {
            step_idx: 0,
            max_steps,
            status: EpisodeStatus::Running,
            transcript: Vec::new(),
        }
    }

    pub fn append(&mut self, line: impl Into<String>) {
        self.transcript.push(line.into());
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            EpisodeStatus::Finished | EpisodeStatus::Truncated | EpisodeStatus::Failed
        )
    }
}
