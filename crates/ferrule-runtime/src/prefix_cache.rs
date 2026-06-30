//! Prefix-aware KV cache — reuse KV blocks across sessions that share token prefixes.
//!
//! Key idea: when multiple requests start with the same prompt (e.g., system
//! messages, few-shot examples), their KV cache can share physical blocks.
//! Uses refcounting for safe eviction.

use super::paged_kv::{BlockId, PagedKvCache};
use std::collections::HashMap;

/// Hash of a token prefix, used as cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixHash(pub u64);

/// Compute a rolling hash of a token sequence.
pub fn hash_tokens(tokens: &[u32]) -> PrefixHash {
    // FNV-1a 64-bit
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    PrefixHash(h)
}

/// A cache entry: a list of physical blocks + refcount.
#[derive(Debug, Clone)]
struct PrefixEntry {
    blocks: Vec<BlockId>,
    /// Number of sessions currently referencing this prefix.
    refcount: usize,
    /// Total tokens in this prefix.
    token_count: usize,
}

/// Prefix-aware KV cache manager.
///
/// Wraps a `PagedKvCache` and adds prefix reuse.
pub struct PrefixCache {
    pool: PagedKvCache,
    /// Prefix hash → cached block chain.
    entries: HashMap<PrefixHash, Vec<PrefixEntry>>,
    /// Per-prefix refcounts.
    max_prefix_len: usize,
    hits: u64,
    misses: u64,
}

impl PrefixCache {
    pub fn new(num_layers: usize, dim: usize, num_blocks: usize, max_prefix_len: usize) -> Self {
        Self {
            pool: PagedKvCache::new(num_layers, dim, num_blocks),
            entries: HashMap::new(),
            max_prefix_len,
            hits: 0,
            misses: 0,
        }
    }

    /// Try to match a prefix in the cache. Returns Some(blocks) on hit,
    /// with refcount incremented. Returns None on miss.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<Vec<BlockId>> {
        for len in (1..=tokens.len().min(self.max_prefix_len)).rev() {
            let prefix = &tokens[..len];
            let hash = hash_tokens(prefix);
            if let Some(entries) = self.entries.get_mut(&hash) {
                // Find the longest matching entry (should be exact match by hash)
                for entry in entries.iter_mut() {
                    if entry.token_count == len {
                        entry.refcount += 1;
                        self.hits += 1;
                        return Some(entry.blocks.clone());
                    }
                }
            }
        }
        self.misses += 1;
        None
    }

    /// Store a completed prefix → blocks mapping in the cache.
    /// The blocks are assumed to already be written with KV data.
    pub fn insert(&mut self, tokens: &[u32], blocks: Vec<BlockId>) {
        if tokens.is_empty() || tokens.len() > self.max_prefix_len {
            return;
        }
        let hash = hash_tokens(tokens);
        let entry = PrefixEntry {
            blocks,
            refcount: 1,
            token_count: tokens.len(),
        };
        self.entries.entry(hash).or_default().push(entry);
    }

    /// Release a reference to a prefix's blocks. When refcount reaches 0,
    /// blocks are freed back to the pool.
    pub fn release(&mut self, tokens: &[u32]) {
        let hash = hash_tokens(tokens);
        if let Some(entries) = self.entries.get_mut(&hash) {
            entries.retain_mut(|entry| {
                if entry.token_count == tokens.len() {
                    entry.refcount = entry.refcount.saturating_sub(1);
                    if entry.refcount == 0 {
                        for &block in &entry.blocks {
                            self.pool.free_block(block);
                        }
                        return false; // remove
                    }
                }
                true
            });
            // Clean up empty entry lists
            if entries.is_empty() {
                self.entries.remove(&hash);
            }
        }
    }

    /// Access the underlying block pool (for reading/writing KV data).
    pub fn pool(&self) -> &PagedKvCache {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut PagedKvCache {
        &mut self.pool
    }

    /// Allocate a new block from the pool.
    pub fn alloc_block(&mut self) -> Option<BlockId> {
        self.pool.alloc_block()
    }

    /// Free a single block.
    pub fn free_block(&mut self, block: BlockId) {
        self.pool.free_block(block);
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic() {
        let h1 = hash_tokens(&[1, 2, 3]);
        let h2 = hash_tokens(&[1, 2, 3]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_differs() {
        let h1 = hash_tokens(&[1, 2, 3]);
        let h2 = hash_tokens(&[1, 2, 4]);
        assert_ne!(h1.0, h2.0);
    }

    #[test]
    fn prefix_cache_hit() {
        let mut cache = PrefixCache::new(1, 2, 10, 32);
        let tokens = vec![1, 2, 3, 4, 5];

        // Miss
        assert!(cache.lookup(&tokens).is_none());

        // Insert blocks (simulate KV write)
        let blocks: Vec<BlockId> = (0..2).map(|_| cache.alloc_block().unwrap()).collect();
        cache.insert(&tokens, blocks.clone());

        // Hit
        let found = cache.lookup(&tokens);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), blocks);

        // Stats
        let (hits, misses) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn prefix_cache_subprefix_match() {
        let mut cache = PrefixCache::new(1, 2, 10, 32);

        let prefix = vec![1, 2];
        let blocks: Vec<BlockId> = (0..1).map(|_| cache.alloc_block().unwrap()).collect();
        cache.insert(&prefix, blocks.clone());

        // Longer sequence that starts with the same prefix should match
        let longer = vec![1, 2, 3, 4];
        let found = cache.lookup(&longer);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), blocks);
    }

    #[test]
    fn prefix_cache_refcount_release() {
        let mut cache = PrefixCache::new(1, 2, 10, 32);
        let tokens = vec![42];

        let b = cache.alloc_block().unwrap();
        cache.insert(&tokens, vec![b]);

        // Two references
        cache.lookup(&tokens); // refcount = 2
        assert_eq!(cache.stats().0, 1);

        // Release one
        cache.release(&tokens);
        // Release second — should free the block
        cache.release(&tokens);

        // Block should be back in pool
        let reused = cache.alloc_block();
        assert_eq!(reused, Some(b));
    }

    #[test]
    fn hit_rate_calculation() {
        let mut cache = PrefixCache::new(1, 2, 4, 4);
        assert_eq!(cache.hit_rate(), 0.0);

        cache.lookup(&[1]); // miss
        cache.lookup(&[1]); // miss
        assert_eq!(cache.hit_rate(), 0.0);
    }
}
