//! KV cache management.
//!
//! - `kv`: contiguous and multi-session KV cache primitives.
//! - `paged_kv`: paged KV cache for memory-efficient long-context serving.
//! - `prefix_cache`: prefix-based KV reuse across requests.
//! - `radix_cache`: radix-tree structured KV cache for maximal sharing.

pub mod kv;
pub mod paged_kv;
pub mod prefix_cache;

pub use paged_kv::{
    BlockId, BlockTable, KvCacheDtype, PagedKvCache, PagedSequenceKvCache, BLOCK_SIZE,
};
pub mod radix_cache;

pub use kv::{
    ContiguousKvCache, KvCache, KvCacheLayout, KvHandle, KvLayerView, MultiSessionKvCache,
    SequenceKvCache,
};
