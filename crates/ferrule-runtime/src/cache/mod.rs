//! KV cache management.
//!
//! - `kv`: contiguous and multi-session KV cache primitives.
//! - `paged_kv`: paged KV cache for memory-efficient long-context serving.
//! - `page_manager`: runtime KV page manager with reserve/commit/rollback.
//! - `prefix_cache`: prefix-based KV reuse across requests.
//! - `radix_cache`: radix-tree structured KV cache for maximal sharing.

pub mod kv;
pub mod page_manager;
pub mod paged_kv;
pub mod prefix_cache;

pub use page_manager::{
    BlockTable as PageBlockTable, KvPageManager, KvPageManagerStats, KvReservationBindings,
    PreemptedKvState, PreparedKvSequenceFork,
};
pub use paged_kv::{
    BLOCK_SIZE, BlockId, BlockTable, KvCacheDtype, PagedKvCache, PagedSequenceKvCache,
};
pub mod radix_cache;

pub use kv::{
    ContiguousKvCache, FixedSequenceSlotPool, KvCache, KvCacheLayout, KvHandle, KvLayerView,
    MultiSessionKvCache, SequenceKvCache, SequenceSlotPool,
};
