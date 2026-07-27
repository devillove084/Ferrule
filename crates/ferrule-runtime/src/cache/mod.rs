//! Authoritative logical KV page transactions.

pub mod page_manager;

pub use page_manager::{
    AbortKvReservationsError, BlockTable as PageBlockTable, ConfirmKvRetirementError,
    KvPageManager, KvPageManagerStats, KvReservation, KvReservationBindings, KvReservationCommit,
    KvReservationId, KvRetirement, PreemptedKvState, PrepareKvCommitError, PreparedKvCommit,
    PreparedKvSequenceFork,
};
