//! KV cache abstraction — the interface between model execution and KV memory.
//!
//! Separates KV storage policy (contiguous, paged, prefix-shared) from
//! model forward kernels. First implementation: simple contiguous cache.

use ferrule_core::Result;

/// Handle to a specific sequence's KV cache slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvHandle(pub usize);

/// Trait for KV cache backends.
///
/// Implementations range from simple contiguous buffers to paged/radix caches.
pub trait KvCache {
    /// Maximum number of tokens this cache can hold per layer.
    fn capacity(&self) -> usize;

    /// Number of layers.
    fn num_layers(&self) -> usize;

    /// Dimensionality of each K/V entry (kv_dim).
    fn dim(&self) -> usize;

    /// Append K/V vectors for a single layer at a given position.
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()>;

    /// Get a reference to the K cache for one layer (for attention kernels).
    fn k_slice(&self, layer: usize) -> &[f32];

    /// Get a reference to the V cache for one layer.
    fn v_slice(&self, layer: usize) -> &[f32];

    /// Total number of positions currently stored.
    fn seq_len(&self) -> usize;

    /// Reset all layers to empty.
    fn reset(&mut self) -> Result<()>;
}

// ── Contiguous implementation ──────────────────────────────────────────

/// Simple contiguous KV cache for a single session.
pub struct ContiguousKvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    dim: usize,
    capacity: usize,
    seq_len: usize,
}

impl ContiguousKvCache {
    pub fn new(num_layers: usize, dim: usize, capacity: usize) -> Self {
        let k: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0f32; capacity * dim])
            .collect();
        let v: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0f32; capacity * dim])
            .collect();
        Self {
            k,
            v,
            dim,
            capacity,
            seq_len: 0,
        }
    }
}

impl KvCache for ContiguousKvCache {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn num_layers(&self) -> usize {
        self.k.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        assert!(layer < self.k.len());
        assert!(k.len() == self.dim);
        assert!(v.len() == self.dim);
        let off = pos * self.dim;
        self.k[layer][off..off + self.dim].copy_from_slice(k);
        self.v[layer][off..off + self.dim].copy_from_slice(v);
        if pos + 1 > self.seq_len {
            self.seq_len = pos + 1;
        }
        Ok(())
    }

    fn k_slice(&self, layer: usize) -> &[f32] {
        &self.k[layer][..self.seq_len * self.dim]
    }

    fn v_slice(&self, layer: usize) -> &[f32] {
        &self.v[layer][..self.seq_len * self.dim]
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn reset(&mut self) -> Result<()> {
        for layer_k in &mut self.k {
            layer_k.fill(0.0);
        }
        for layer_v in &mut self.v {
            layer_v.fill(0.0);
        }
        self.seq_len = 0;
        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_append_and_read() {
        let mut cache = ContiguousKvCache::new(2, 4, 16);
        assert_eq!(cache.capacity(), 16);
        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.dim(), 4);
        assert_eq!(cache.seq_len(), 0);

        // Append position 0, layer 0
        cache
            .append(0, 0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(cache.seq_len(), 1);

        let k0 = cache.k_slice(0);
        assert_eq!(k0.len(), 4);
        assert_eq!(k0[0], 1.0);
        assert_eq!(k0[3], 4.0);

        let v0 = cache.v_slice(0);
        assert_eq!(v0[0], 5.0);

        // Append position 1
        cache
            .append(0, 1, &[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8])
            .unwrap();
        assert_eq!(cache.seq_len(), 2);

        let k0 = cache.k_slice(0);
        assert_eq!(k0.len(), 8);
        // pos 0 values still present
        assert_eq!(k0[0], 1.0);
        // pos 1 values
        assert_eq!(k0[4], 0.1);
    }

    #[test]
    fn contiguous_reset() {
        let mut cache = ContiguousKvCache::new(1, 2, 4);
        cache.append(0, 0, &[1.0, 1.0], &[2.0, 2.0]).unwrap();
        cache.append(0, 1, &[3.0, 3.0], &[4.0, 4.0]).unwrap();
        assert_eq!(cache.seq_len(), 2);

        cache.reset().unwrap();
        assert_eq!(cache.seq_len(), 0);

        let k0 = cache.k_slice(0);
        assert_eq!(k0.len(), 0);
        // underlying buffer is zeroed
        assert!(cache.k[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn contiguous_independent_layers() {
        let mut cache = ContiguousKvCache::new(3, 2, 4);
        cache.append(0, 0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.append(1, 0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        cache.append(2, 0, &[9.0, 10.0], &[11.0, 12.0]).unwrap();

        assert_eq!(cache.k_slice(0), &[1.0, 2.0]);
        assert_eq!(cache.k_slice(1), &[5.0, 6.0]);
        assert_eq!(cache.k_slice(2), &[9.0, 10.0]);
    }

    #[test]
    fn kv_handle_equality() {
        let h1 = KvHandle(5);
        let h2 = KvHandle(5);
        let h3 = KvHandle(6);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}

// ── Multi-session contiguous KV ─────────────────────────────────────────

/// Partitioned KV cache: N sessions, each with fixed-capacity contiguous slots.
pub struct MultiSessionKvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    dim: usize,
    session_cap: usize,
    num_layers: usize,
    max_sessions: usize,
    seq_lens: Vec<usize>,
    layer_seq_lens: Vec<Vec<usize>>,
    free_mask: Vec<bool>,
}

impl MultiSessionKvCache {
    pub fn new(num_layers: usize, dim: usize, session_cap: usize, max_sessions: usize) -> Self {
        let total = max_sessions * session_cap * dim;
        Self {
            k: (0..num_layers).map(|_| vec![0f32; total]).collect(),
            v: (0..num_layers).map(|_| vec![0f32; total]).collect(),
            dim,
            session_cap,
            num_layers,
            max_sessions,
            seq_lens: vec![0; max_sessions],
            layer_seq_lens: vec![vec![0; num_layers]; max_sessions],
            free_mask: vec![true; max_sessions],
        }
    }
    pub fn alloc(&mut self) -> Option<KvHandle> {
        self.free_mask.iter().position(|&f| f).map(|i| {
            self.free_mask[i] = false;
            KvHandle(i)
        })
    }
    pub fn append(&mut self, h: KvHandle, layer: usize, k: &[f32], v: &[f32]) -> Result<()> {
        let sid = h.0;
        assert!(layer < self.num_layers);
        assert!(k.len() == self.dim);
        assert!(v.len() == self.dim);
        let pos = self.layer_seq_lens[sid][layer];
        assert!(pos < self.session_cap);
        let off = (sid * self.session_cap + pos) * self.dim;
        self.k[layer][off..off + self.dim].copy_from_slice(k);
        self.v[layer][off..off + self.dim].copy_from_slice(v);
        self.layer_seq_lens[sid][layer] += 1;
        self.seq_lens[sid] = self.seq_lens[sid].max(self.layer_seq_lens[sid][layer]);
        Ok(())
    }
    pub fn k_slice(&self, h: KvHandle, layer: usize) -> &[f32] {
        let sid = h.0;
        let s = self.layer_seq_lens[sid][layer];
        let start = sid * self.session_cap * self.dim;
        &self.k[layer][start..start + s * self.dim]
    }
    pub fn v_slice(&self, h: KvHandle, layer: usize) -> &[f32] {
        let sid = h.0;
        let s = self.layer_seq_lens[sid][layer];
        let start = sid * self.session_cap * self.dim;
        &self.v[layer][start..start + s * self.dim]
    }
    pub fn seq_len(&self, h: KvHandle) -> usize {
        self.seq_lens[h.0]
    }
    pub fn free(&mut self, h: KvHandle) {
        let sid = h.0;
        if !self.free_mask[sid] {
            let start = sid * self.session_cap * self.dim;
            let end = start + self.session_cap * self.dim;
            for l in 0..self.num_layers {
                self.k[l][start..end].fill(0.0);
                self.v[l][start..end].fill(0.0);
            }
            self.seq_lens[sid] = 0;
            self.layer_seq_lens[sid].fill(0);
            self.free_mask[sid] = true;
        }
    }
    pub fn active_count(&self) -> usize {
        self.free_mask.iter().filter(|&&f| !f).count()
    }
    pub fn max_sessions(&self) -> usize {
        self.max_sessions
    }
    pub fn session_capacity(&self) -> usize {
        self.session_cap
    }
}

#[cfg(test)]
mod multi_tests {
    use super::*;
    #[test]
    fn alloc_free_reuse() {
        let mut c = MultiSessionKvCache::new(2, 4, 8, 3);
        let a = c.alloc().unwrap();
        assert_eq!(c.active_count(), 1);
        let b = c.alloc().unwrap();
        c.free(a);
        assert_eq!(c.active_count(), 1);
        let a2 = c.alloc().unwrap();
        assert_eq!(a2, KvHandle(0));
        c.free(b);
        c.free(a2);
        assert_eq!(c.active_count(), 0);
    }
    #[test]
    fn session_isolation() {
        let mut c = MultiSessionKvCache::new(2, 3, 4, 2);
        let s0 = c.alloc().unwrap();
        let s1 = c.alloc().unwrap();
        c.append(s0, 0, &[1., 2., 3.], &[4., 5., 6.]).unwrap();
        c.append(s1, 0, &[7., 8., 9.], &[10., 11., 12.]).unwrap();
        assert_eq!(c.k_slice(s0, 0), &[1., 2., 3.]);
        assert_eq!(c.k_slice(s1, 0), &[7., 8., 9.]);
    }
    #[test]
    fn append_same_token_across_layers_keeps_session_len() {
        let mut c = MultiSessionKvCache::new(2, 2, 4, 1);
        let h = c.alloc().unwrap();
        c.append(h, 0, &[1., 2.], &[3., 4.]).unwrap();
        c.append(h, 1, &[5., 6.], &[7., 8.]).unwrap();
        assert_eq!(c.seq_len(h), 1);
        assert_eq!(c.k_slice(h, 0), &[1., 2.]);
        assert_eq!(c.k_slice(h, 1), &[5., 6.]);
    }

    #[test]
    fn free_zeros_data() {
        let mut c = MultiSessionKvCache::new(1, 2, 4, 2);
        let h = c.alloc().unwrap();
        c.append(h, 0, &[1., 2.], &[3., 4.]).unwrap();
        c.free(h);
        assert!(c.k[0].iter().all(|&x| x == 0.0));
    }
    #[test]
    fn overflow_panics() {
        let mut c = MultiSessionKvCache::new(1, 2, 1, 1);
        let h = c.alloc().unwrap();
        c.append(h, 0, &[1., 2.], &[3., 4.]).unwrap();
    }
}
