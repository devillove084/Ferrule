//! Fixed-size owner-local observations, never resource or completion authority.
use std::time::{Duration, Instant};

pub(super) fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[derive(Default)]
pub(super) struct HostCallTiming {
    pub calls: u64,
    pub failures: u64,
    pub elapsed: Duration,
}

impl HostCallTiming {
    pub fn measure<T, E>(&mut self, call: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
        let start = Instant::now();
        let result = call();
        self.calls += 1;
        self.failures += u64::from(result.is_err());
        self.elapsed += start.elapsed();
        result
    }
}

#[derive(Default)]
pub(super) struct ProofTiming {
    // Event creation/record plus existing synchronization, not device event time.
    pub consumer_event: HostCallTiming,
    pub compute_sync: HostCallTiming,
    pub upload_sync: HostCallTiming,
}

macro_rules! counters {
    ($($field:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Default)]
        pub(super) struct CounterSnapshot { $(pub $field: u64,)+ }
        pub(super) struct CounterDelta { $(pub $field: Option<u64>,)+ }
        impl CounterSnapshot {
            pub fn delta(self, before: Self) -> CounterDelta {
                CounterDelta { $($field: self.$field.checked_sub(before.$field),)+ }
            }
        }
        impl CounterDelta {
            pub fn valid(&self) -> bool { $(self.$field.is_some())&&+ }
        }
    }
}

counters!(
    kernel_launches,
    h2d_copies,
    h2d_bytes,
    d2h_copies,
    d2h_bytes,
    artifact_uploads,
    artifact_upload_bytes,
    allocation_attempts,
    allocation_bytes,
    stream_syncs,
    stream_sync_failures,
    expert_hits,
    expert_uploads,
    expert_evictions,
    read_calls,
    read_bytes,
    source_checks,
    path_metadata_calls,
    handle_metadata_calls,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_reset_is_missing_delta_not_zero() {
        let before = CounterSnapshot {
            expert_hits: 8,
            read_bytes: 100,
            ..Default::default()
        };
        let after = CounterSnapshot {
            expert_hits: 2,
            read_bytes: 120,
            ..Default::default()
        };
        let delta = after.delta(before);
        assert!(!delta.valid());
        assert_eq!(delta.expert_hits, None);
        assert_eq!(delta.read_bytes, Some(20));
        let unchanged = before.delta(before);
        assert!(unchanged.valid());
        assert_eq!(unchanged.expert_hits, Some(0));
    }

    #[test]
    fn timing_preserves_result_and_counts_failure() {
        let mut proofs = ProofTiming::default();
        assert_eq!(proofs.consumer_event.measure(|| Ok::<_, &str>(7)), Ok(7));
        assert_eq!(
            proofs.consumer_event.measure(|| Err::<(), _>("unknown")),
            Err("unknown")
        );
        assert_eq!(proofs.consumer_event.calls, 2);
        assert_eq!(proofs.consumer_event.failures, 1);
        assert_eq!(proofs.compute_sync.calls, 0);
        assert_eq!(proofs.upload_sync.calls, 0);
        assert_eq!(micros(Duration::from_nanos(1_999)), 1);
    }
}
