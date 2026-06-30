//! Radix cache — radix tree over token prefixes for KV page sharing.
//!
//! SGLang-style: a radix tree maps token prefix paths to cached KV blocks.
//! When a new request shares a prefix with a previously cached sequence,
//! the matching KV blocks can be reused, skipping the prefill for that prefix.
//!
//! Each node stores token → child edges, optional cached blocks, and a
//! reference count for safe eviction.

use super::paged_kv::BlockId;
use std::collections::HashMap;

/// A node in the radix tree.
struct RadixNode {
    /// Token → child node edges.
    children: HashMap<u32, RadixNode>,
    /// KV blocks for the prefix ending at this node, if cached.
    blocks: Option<Vec<BlockId>>,
    /// Reference count for shared blocks (active sessions using this prefix).
    refcount: usize,
}

impl RadixNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            blocks: None,
            refcount: 0,
        }
    }
}

/// Radix tree cache for KV block reuse across sessions.
///
/// Tracks hits and misses to compute a hit rate.
pub struct RadixCache {
    root: RadixNode,
    hits: u64,
    misses: u64,
}

impl RadixCache {
    /// Create an empty radix cache.
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Insert a token prefix → KV blocks mapping into the cache.
    ///
    /// Walks the radix tree, creating missing nodes along the path,
    /// and stores the blocks at the terminal node. Sets refcount to 1.
    pub fn insert(&mut self, tokens: &[u32], blocks: Vec<BlockId>) {
        if tokens.is_empty() {
            return;
        }
        let mut node = &mut self.root;
        for &token in tokens {
            node = node.children.entry(token).or_insert_with(RadixNode::new);
        }
        node.blocks = Some(blocks);
        node.refcount = 1;
    }

    /// Look up the longest matching prefix in the cache.
    ///
    /// Walks the radix tree token-by-token, tracking the last node
    /// that has cached blocks. Returns `(match_len, Some(blocks))` on hit
    /// and increments the refcount, or `(0, None)` on complete miss.
    pub fn lookup(&mut self, tokens: &[u32]) -> (usize, Option<Vec<BlockId>>) {
        // First pass: find the longest prefix match (read-only walk).
        let mut match_len = 0;
        {
            let mut node = &self.root;
            for (i, &token) in tokens.iter().enumerate() {
                match node.children.get(&token) {
                    Some(child) => {
                        node = child;
                        if node.blocks.is_some() {
                            match_len = i + 1;
                        }
                    }
                    None => break,
                }
            }
        }

        if match_len == 0 {
            self.misses += 1;
            return (0, None);
        }

        // Second pass: walk to the match node mutably to increment refcount.
        let prefix = &tokens[..match_len];
        let mut node = &mut self.root;
        for &token in prefix {
            node = node.children.get_mut(&token).unwrap();
        }
        node.refcount += 1;
        self.hits += 1;
        (match_len, node.blocks.clone())
    }

    /// Release a reference to a token prefix.
    ///
    /// Decrements the refcount on the matching node. When refcount
    /// reaches 0, the cached blocks are dropped (indicating they can
    /// be freed back to the pool).
    pub fn release(&mut self, tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        let mut node = &mut self.root;
        for &token in tokens {
            match node.children.get_mut(&token) {
                Some(child) => node = child,
                None => return, // prefix not found
            }
        }
        if node.blocks.is_some() {
            node.refcount = node.refcount.saturating_sub(1);
            if node.refcount == 0 {
                // Drop the cached blocks — caller should also free actual blocks.
                node.blocks = None;
            }
        }
    }

    /// Return the hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Return (hits, misses) counts.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

impl Default for RadixCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bid(id: usize) -> BlockId {
        BlockId(id)
    }

    #[test]
    fn radix_cache_empty_lookup() {
        let mut cache = RadixCache::new();
        let (len, blocks) = cache.lookup(&[1, 2, 3]);
        assert_eq!(len, 0);
        assert!(blocks.is_none());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn radix_cache_insert_and_hit() {
        let mut cache = RadixCache::new();
        let tokens = vec![1, 2, 3];
        let blocks = vec![bid(0), bid(1)];

        cache.insert(&tokens, blocks.clone());

        let (len, found) = cache.lookup(&tokens);
        assert_eq!(len, 3);
        assert_eq!(found, Some(blocks));
        assert_eq!(cache.stats().0, 1); // 1 hit
    }

    #[test]
    fn radix_cache_longest_prefix_match() {
        let mut cache = RadixCache::new();

        // Insert prefix [1, 2]
        cache.insert(&[1, 2], vec![bid(0)]);

        // Insert longer prefix [1, 2, 3]
        cache.insert(&[1, 2, 3], vec![bid(0), bid(1)]);

        // Lookup [1, 2, 3, 4] — should match [1, 2, 3] (length 3)
        let (len, found) = cache.lookup(&[1, 2, 3, 4]);
        assert_eq!(len, 3);
        assert_eq!(found, Some(vec![bid(0), bid(1)]));
    }

    #[test]
    fn radix_cache_partial_match() {
        let mut cache = RadixCache::new();

        // Insert [1, 2]
        cache.insert(&[1, 2], vec![bid(0)]);

        // Lookup [1, 2, 3] — should match [1, 2] (length 2)
        let (len, found) = cache.lookup(&[1, 2, 3]);
        assert_eq!(len, 2);
        assert_eq!(found, Some(vec![bid(0)]));
    }

    #[test]
    fn radix_cache_no_match_different_path() {
        let mut cache = RadixCache::new();

        // Insert [1, 2, 3]
        cache.insert(&[1, 2, 3], vec![bid(0)]);

        // Lookup [1, 2, 4] — no match at all
        let (len, found) = cache.lookup(&[1, 2, 4]);
        assert_eq!(len, 0);
        assert!(found.is_none());
    }

    #[test]
    fn radix_cache_release_and_refcount() {
        let mut cache = RadixCache::new();
        let tokens = vec![5, 6];
        let blocks = vec![bid(10)];

        cache.insert(&tokens, blocks.clone());

        // First lookup: refcount goes to 2
        let (len, _) = cache.lookup(&tokens);
        assert_eq!(len, 2);

        // Release one reference
        cache.release(&tokens);
        // Should still be cached (refcount > 0)
        let (len2, found2) = cache.lookup(&tokens);
        assert_eq!(len2, 2);
        assert!(found2.is_some());

        // Release twice more to bring refcount to 0
        cache.release(&tokens);
        cache.release(&tokens);
        // After refcount hits 0, blocks are dropped
        let (len3, found3) = cache.lookup(&tokens);
        // The node still exists but blocks are None
        assert_eq!(len3, 0);
        assert!(found3.is_none());
    }

    #[test]
    fn radix_cache_hit_rate() {
        let mut cache = RadixCache::new();

        // 2 misses
        cache.lookup(&[1]);
        cache.lookup(&[2]);
        assert_eq!(cache.hit_rate(), 0.0);

        // Insert and get 1 hit
        cache.insert(&[3], vec![bid(0)]);
        cache.lookup(&[3]);
        assert!(cache.hit_rate() > 0.0);
        assert_eq!(cache.stats(), (1, 2));
    }

    #[test]
    fn radix_cache_overlapping_prefixes() {
        let mut cache = RadixCache::new();

        // Insert two overlapping prefixes
        cache.insert(&[1, 2], vec![bid(0)]);
        cache.insert(&[1, 2, 3], vec![bid(0), bid(1)]);

        // Shorter prefix lookup
        let (len1, found1) = cache.lookup(&[1, 2]);
        assert_eq!(len1, 2);
        assert_eq!(found1, Some(vec![bid(0)]));

        // Longer prefix lookup (different request)
        let (len2, found2) = cache.lookup(&[1, 2, 3, 4]);
        assert_eq!(len2, 3);
        assert_eq!(found2, Some(vec![bid(0), bid(1)]));
    }
}
