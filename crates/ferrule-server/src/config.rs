use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ferrule_model::ChatTemplate;

#[derive(Debug, Clone)]
pub struct ModelRegistration {
    pub id: String,
    pub owned_by: String,
    pub created: u64,
    pub chat_template: ChatTemplate,
}

impl ModelRegistration {
    pub fn new(id: impl Into<String>, chat_template: ChatTemplate) -> Self {
        Self {
            id: id.into(),
            owned_by: "ferrule".into(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            chat_template,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Commands waiting to be accepted by the single model owner.
    pub command_queue_capacity: usize,
    /// Token and terminal events buffered independently for each request.
    pub event_queue_capacity: usize,
    /// Maximum commands handled between model steps so request floods cannot starve decode.
    pub max_commands_per_tick: usize,
    /// Backoff when the runtime reports work but cannot currently schedule it.
    pub blocked_backoff: Duration,
    /// Maximum time an HTTP handler waits for tokenization and runtime admission.
    pub admission_timeout: Duration,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            command_queue_capacity: 256,
            event_queue_capacity: 32,
            max_commands_per_tick: 64,
            blocked_backoff: Duration::from_millis(1),
            admission_timeout: Duration::from_secs(30),
        }
    }
}

impl WorkerConfig {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.command_queue_capacity == 0 {
            return Err("command_queue_capacity must be greater than zero");
        }
        if self.event_queue_capacity < 2 {
            return Err("event_queue_capacity must be at least two");
        }
        if self.max_commands_per_tick == 0 {
            return Err("max_commands_per_tick must be greater than zero");
        }
        if self.admission_timeout.is_zero() {
            return Err("admission_timeout must be greater than zero");
        }
        Ok(())
    }
}
