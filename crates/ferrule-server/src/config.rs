use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ferrule_common::ServingConfigError;
use ferrule_model::ChatTemplate;
use ferrule_runtime::RuntimeAdmissionOptions;

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
    /// Maximum time an HTTP handler waits for tokenization and runtime admission.
    pub admission_timeout: Duration,
    /// One absolute cooperative budget for cancellation, physical close, and host join.
    /// Default: 30 seconds. Callers may configure it; zero requests immediate
    /// incomplete reporting. This cannot kill a thread or interrupt a native call.
    pub shutdown_timeout: Duration,
    /// Maximum concurrent server request leases, including slow responses and
    /// terminal requests whose runtime cleanup has not yet been observed.
    pub max_inflight_requests: usize,
    /// Maximum UTF-8 prompt/body bytes held by the server admission ledger.
    pub max_prompt_bytes: usize,
    /// Per-HTTP-body bound enforced while reading, before JSON/tokenization.
    pub max_body_bytes: usize,
    /// Optional runtime-owned admission limits. `None` preserves the engine's
    /// own defaults and keeps legacy engines compatible.
    pub runtime_admission_options: Option<RuntimeAdmissionOptions>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            command_queue_capacity: 256,
            event_queue_capacity: 32,
            max_commands_per_tick: 64,
            admission_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(30),
            max_inflight_requests: 1024,
            max_prompt_bytes: 64 * 1024 * 1024,
            max_body_bytes: 16 * 1024 * 1024,
            runtime_admission_options: None,
        }
    }
}

impl WorkerConfig {
    pub fn validate_admission(&self) -> ferrule_runtime::Result<()> {
        if self.max_inflight_requests == 0 || self.max_prompt_bytes == 0 || self.max_body_bytes == 0
        {
            return Err(ferrule_runtime::Error::InvalidRequest {
                message: "server admission request, prompt-byte and body limits must be nonzero"
                    .into(),
            });
        }
        // Byte lengths and doubled JSON reservations must remain representable.
        // Queue sizes also have Tokio's stricter semaphore limit.
        if self.max_inflight_requests > isize::MAX as usize
            || self.max_prompt_bytes > isize::MAX as usize
            || self.max_body_bytes > isize::MAX as usize
            || self.command_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
            || self.event_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
            || std::time::Instant::now()
                .checked_add(self.admission_timeout)
                .is_none()
            || std::time::Instant::now()
                .checked_add(self.shutdown_timeout)
                .is_none()
        {
            return Err(ferrule_runtime::Error::InvalidRequest {
                message:
                    "server admission limits, queue capacity or timeout exceed supported range"
                        .into(),
            });
        }
        if let Some(options) = self.runtime_admission_options {
            if [
                options.max_waiting_requests,
                options.max_request_identities,
                options.max_session_identities,
            ]
            .into_iter()
            .any(|limit| limit > isize::MAX as usize)
            {
                return Err(ferrule_runtime::RuntimeAdmissionError::InvalidOptions.into());
            }
            options.validate()?;
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), ServingConfigError> {
        if self.command_queue_capacity == 0 {
            return Err(ServingConfigError::ZeroCommandQueueCapacity);
        }
        if self.event_queue_capacity < 2 {
            return Err(ServingConfigError::EventQueueCapacityTooSmall {
                actual: self.event_queue_capacity,
            });
        }
        if self.max_commands_per_tick == 0 {
            return Err(ServingConfigError::ZeroCommandsPerTick);
        }
        if self.admission_timeout.is_zero() {
            return Err(ServingConfigError::ZeroAdmissionTimeout);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_config_rejects_zero_overflow_and_waiting_identity_boundary() {
        let mut config = WorkerConfig::default();
        config.validate_admission().unwrap();
        for field in 0..6 {
            for value in [0, usize::MAX] {
                let mut invalid = config.clone();
                let mut runtime = RuntimeAdmissionOptions::default();
                match field {
                    0 => invalid.max_inflight_requests = value,
                    1 => invalid.max_prompt_bytes = value,
                    2 => invalid.max_body_bytes = value,
                    3 => runtime.max_waiting_requests = value,
                    4 => runtime.max_request_identities = value,
                    5 => runtime.max_session_identities = value,
                    _ => unreachable!(),
                }
                invalid.runtime_admission_options = Some(runtime);
                assert!(
                    invalid.validate_admission().is_err(),
                    "field={field}, value={value}"
                );
            }
        }
        config.runtime_admission_options = Some(RuntimeAdmissionOptions {
            max_waiting_requests: 2,
            max_request_identities: 1,
            max_session_identities: 1,
        });
        assert!(config.validate_admission().is_err());
        config
            .runtime_admission_options
            .as_mut()
            .unwrap()
            .max_request_identities = 2;
        // Session identities, HTTP leases and body/aggregate bytes are separate
        // capacities; smaller limits intentionally apply backpressure.
        config.max_inflight_requests = 100;
        config.max_prompt_bytes = 1;
        config.max_body_bytes = 100;
        config.validate_admission().unwrap();
        for command in [true, false] {
            let mut invalid = config.clone();
            if command {
                invalid.command_queue_capacity = usize::MAX;
            } else {
                invalid.event_queue_capacity = usize::MAX;
            }
            assert!(invalid.validate_admission().is_err());
        }
    }
}
