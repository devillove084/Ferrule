//! KV cache abstraction — the interface between model execution and KV memory.
//!
//! Separates KV storage policy (contiguous, paged, prefix-shared) from
//! model forward kernels. First implementation: simple contiguous cache.

use ferrule_core::{Error, Result};

/// Handle to a specific sequence's KV cache slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvHandle(pub usize);

/// Physical KV storage layout.
///
/// The runtime should reason in terms of logical sequences and layer views;
/// concrete backends are free to store those views contiguously, in fixed
/// session partitions, or in vLLM-style pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheLayout {
    /// One contiguous sequence: `[layer][seq][kv_dim]`.
    SingleContiguous,
    /// Fixed slots for multiple contiguous sequences.
    MultiContiguous { max_sessions: usize },
    /// Fixed-size block table. `block_size` is measured in tokens.
    Paged { block_size: usize },
}

/// Read-only host view for one layer of one logical sequence.
#[derive(Debug, Clone, Copy)]
pub struct KvLayerView<'a> {
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub seq_len: usize,
    pub dim: usize,
}

impl<'a> KvLayerView<'a> {
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn validate(&self) -> Result<()> {
        let expected = self
            .seq_len
            .checked_mul(self.dim)
            .ok_or_else(|| Error::Internal("KV layer view length overflow".into()))?;
        if self.k.len() != expected || self.v.len() != expected {
            return Err(Error::Internal(format!(
                "KV view has k={} v={} floats, expected {}",
                self.k.len(),
                self.v.len(),
                expected
            )));
        }
        Ok(())
    }
}

/// Sequence-oriented KV cache API.
///
/// This is the boundary Ferrule should grow toward for vLLM/SGLang-like serving:
/// sequence ownership is explicit, while attention kernels receive a per-layer
/// view appropriate for the concrete backend.
pub trait SequenceKvCache {
    fn storage_layout(&self) -> KvCacheLayout;
    fn layer_count(&self) -> usize;
    fn entry_dim(&self) -> usize;
    fn sequence_capacity(&self, handle: KvHandle) -> usize;
    fn alloc_sequence(&mut self) -> Result<KvHandle>;
    fn free_sequence(&mut self, handle: KvHandle) -> Result<()>;
    fn append_to_sequence(
        &mut self,
        handle: KvHandle,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()>;
    fn layer_view(&self, handle: KvHandle, layer: usize) -> Result<KvLayerView<'_>>;
    fn sequence_len(&self, handle: KvHandle) -> usize;
}

/// Trait for legacy single-sequence KV cache backends.
///
/// Implementations range from simple contiguous buffers to paged/radix caches.
pub trait KvCache {
    /// Physical storage layout.
    fn layout(&self) -> KvCacheLayout;

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

    /// Get K/V together as an explicit view for attention code.
    fn layer_view(&self, layer: usize) -> KvLayerView<'_>;

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

    fn validate_write(&self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if layer >= self.k.len() {
            return Err(Error::Internal(format!("KV layer {layer} out of range")));
        }
        if pos >= self.capacity {
            return Err(Error::Internal(format!(
                "KV position {pos} exceeds capacity {}",
                self.capacity
            )));
        }
        if k.len() != self.dim || v.len() != self.dim {
            return Err(Error::Internal(format!(
                "KV dim mismatch: k={} v={} expected {}",
                k.len(),
                v.len(),
                self.dim
            )));
        }
        Ok(())
    }
}

impl KvCache for ContiguousKvCache {
    fn layout(&self) -> KvCacheLayout {
        KvCacheLayout::SingleContiguous
    }
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
        self.validate_write(layer, pos, k, v)?;
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

    fn layer_view(&self, layer: usize) -> KvLayerView<'_> {
        KvLayerView {
            k: self.k_slice(layer),
            v: self.v_slice(layer),
            seq_len: self.seq_len,
            dim: self.dim,
        }
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

impl SequenceKvCache for ContiguousKvCache {
    fn storage_layout(&self) -> KvCacheLayout {
        KvCacheLayout::SingleContiguous
    }

    fn layer_count(&self) -> usize {
        self.k.len()
    }

    fn entry_dim(&self) -> usize {
        self.dim
    }

    fn sequence_capacity(&self, handle: KvHandle) -> usize {
        if handle == KvHandle(0) {
            self.capacity
        } else {
            0
        }
    }

    fn alloc_sequence(&mut self) -> Result<KvHandle> {
        self.reset()?;
        Ok(KvHandle(0))
    }

    fn free_sequence(&mut self, handle: KvHandle) -> Result<()> {
        if handle != KvHandle(0) {
            return Err(Error::Internal(format!("unknown KV handle {}", handle.0)));
        }
        self.reset()
    }

    fn append_to_sequence(
        &mut self,
        handle: KvHandle,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        if handle != KvHandle(0) {
            return Err(Error::Internal(format!("unknown KV handle {}", handle.0)));
        }
        self.append(layer, pos, k, v)
    }

    fn layer_view(&self, handle: KvHandle, layer: usize) -> Result<KvLayerView<'_>> {
        if handle != KvHandle(0) {
            return Err(Error::Internal(format!("unknown KV handle {}", handle.0)));
        }
        if layer >= self.k.len() {
            return Err(Error::Internal(format!("KV layer {layer} out of range")));
        }
        Ok(KvCache::layer_view(self, layer))
    }

    fn sequence_len(&self, handle: KvHandle) -> usize {
        if handle == KvHandle(0) {
            self.seq_len
        } else {
            0
        }
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

    #[test]
    fn contiguous_sequence_api_returns_layer_view() {
        let mut cache = ContiguousKvCache::new(2, 2, 4);
        let handle = SequenceKvCache::alloc_sequence(&mut cache).unwrap();
        assert_eq!(
            SequenceKvCache::storage_layout(&cache),
            KvCacheLayout::SingleContiguous
        );
        SequenceKvCache::append_to_sequence(&mut cache, handle, 0, 0, &[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        let view = SequenceKvCache::layer_view(&cache, handle, 0).unwrap();
        view.validate().unwrap();
        assert_eq!(view.seq_len, 1);
        assert_eq!(view.k, &[1.0, 2.0]);
        assert_eq!(view.v, &[3.0, 4.0]);
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

    fn validate_handle(&self, h: KvHandle) -> Result<usize> {
        let sid = h.0;
        if sid >= self.max_sessions || self.free_mask[sid] {
            return Err(Error::Internal(format!("inactive KV handle {sid}")));
        }
        Ok(sid)
    }

    fn validate_write(
        &self,
        sid: usize,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        if layer >= self.num_layers {
            return Err(Error::Internal(format!("KV layer {layer} out of range")));
        }
        if pos >= self.session_cap {
            return Err(Error::Internal(format!(
                "KV position {pos} exceeds session capacity {}",
                self.session_cap
            )));
        }
        if k.len() != self.dim || v.len() != self.dim {
            return Err(Error::Internal(format!(
                "KV dim mismatch: k={} v={} expected {}",
                k.len(),
                v.len(),
                self.dim
            )));
        }
        if sid >= self.max_sessions {
            return Err(Error::Internal(format!("KV session {sid} out of range")));
        }
        Ok(())
    }

    pub fn append(&mut self, h: KvHandle, layer: usize, k: &[f32], v: &[f32]) -> Result<()> {
        let sid = self.validate_handle(h)?;
        if layer >= self.num_layers {
            return Err(Error::Internal(format!("KV layer {layer} out of range")));
        }
        let pos = self.layer_seq_lens[sid][layer];
        self.validate_write(sid, layer, pos, k, v)?;
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
    pub fn layer_view(&self, h: KvHandle, layer: usize) -> Result<KvLayerView<'_>> {
        let sid = self.validate_handle(h)?;
        if layer >= self.num_layers {
            return Err(Error::Internal(format!("KV layer {layer} out of range")));
        }
        let seq_len = self.layer_seq_lens[sid][layer];
        let view = KvLayerView {
            k: self.k_slice(h, layer),
            v: self.v_slice(h, layer),
            seq_len,
            dim: self.dim,
        };
        view.validate()?;
        Ok(view)
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

impl SequenceKvCache for MultiSessionKvCache {
    fn storage_layout(&self) -> KvCacheLayout {
        KvCacheLayout::MultiContiguous {
            max_sessions: self.max_sessions,
        }
    }

    fn layer_count(&self) -> usize {
        self.num_layers
    }

    fn entry_dim(&self) -> usize {
        self.dim
    }

    fn sequence_capacity(&self, handle: KvHandle) -> usize {
        if handle.0 < self.max_sessions && !self.free_mask[handle.0] {
            self.session_cap
        } else {
            0
        }
    }

    fn alloc_sequence(&mut self) -> Result<KvHandle> {
        self.alloc()
            .ok_or_else(|| Error::Internal("no free KV session slots".into()))
    }

    fn free_sequence(&mut self, handle: KvHandle) -> Result<()> {
        self.validate_handle(handle)?;
        self.free(handle);
        Ok(())
    }

    fn append_to_sequence(
        &mut self,
        handle: KvHandle,
        layer: usize,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        let sid = self.validate_handle(handle)?;
        self.validate_write(sid, layer, pos, k, v)?;
        let off = (sid * self.session_cap + pos) * self.dim;
        self.k[layer][off..off + self.dim].copy_from_slice(k);
        self.v[layer][off..off + self.dim].copy_from_slice(v);
        self.layer_seq_lens[sid][layer] = self.layer_seq_lens[sid][layer].max(pos + 1);
        self.seq_lens[sid] = self.seq_lens[sid].max(self.layer_seq_lens[sid][layer]);
        Ok(())
    }

    fn layer_view(&self, handle: KvHandle, layer: usize) -> Result<KvLayerView<'_>> {
        self.layer_view(handle, layer)
    }

    fn sequence_len(&self, handle: KvHandle) -> usize {
        if handle.0 < self.max_sessions && !self.free_mask[handle.0] {
            self.seq_lens[handle.0]
        } else {
            0
        }
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
    fn overflow_returns_error() {
        let mut c = MultiSessionKvCache::new(1, 2, 1, 1);
        let h = c.alloc().unwrap();
        c.append(h, 0, &[1., 2.], &[3., 4.]).unwrap();
        assert!(c.append(h, 0, &[5., 6.], &[7., 8.]).is_err());
    }

    #[test]
    fn multi_sequence_api_returns_isolated_views() {
        let mut c = MultiSessionKvCache::new(2, 2, 4, 2);
        let s0 = SequenceKvCache::alloc_sequence(&mut c).unwrap();
        let s1 = SequenceKvCache::alloc_sequence(&mut c).unwrap();
        assert_eq!(
            SequenceKvCache::storage_layout(&c),
            KvCacheLayout::MultiContiguous { max_sessions: 2 }
        );

        SequenceKvCache::append_to_sequence(&mut c, s0, 0, 0, &[1., 2.], &[3., 4.]).unwrap();
        SequenceKvCache::append_to_sequence(&mut c, s1, 0, 0, &[10., 20.], &[30., 40.]).unwrap();

        let v0 = SequenceKvCache::layer_view(&c, s0, 0).unwrap();
        let v1 = SequenceKvCache::layer_view(&c, s1, 0).unwrap();
        assert_eq!(v0.k, &[1., 2.]);
        assert_eq!(v1.k, &[10., 20.]);
        assert_eq!(SequenceKvCache::sequence_len(&c, s0), 1);
        assert_eq!(SequenceKvCache::sequence_len(&c, s1), 1);
    }
}
