//! Paged KV cache — block-based allocation with free list.
//!
//! Based on vLLM's PagedAttention: KV is stored in fixed-size blocks (pages).
//! Each session has a block table mapping logical positions → physical blocks.
//! Attention kernels read the block table to gather K/V.

use ferrule_common::{Error, Result};

use super::kv::{KvCacheLayout, KvHandle, KvLayerView, SequenceKvCache};

/// KV cache dtype policy.
#[derive(Debug, Clone, Copy)]
pub enum KvCacheDtype {
    /// Full FP32 (default, highest quality).
    Fp32,
    /// FP16 (half memory, very small quality loss).
    Fp16,
    /// Int8 with per-token scaling (KIVI-style, 4x compression).
    Int8,
    /// Int4 with per-group scaling (aggressive, 8x compression).
    Int4,
}

impl KvCacheDtype {
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 => 2,
            Self::Int8 => 1,
            Self::Int4 => 1, // 0.5 actually, but rounded
        }
    }
}

/// A physical block index (address in the KV pool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockId(pub usize);

/// Block table: maps logical token positions to physical blocks.
/// Each session owns one block table.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Ordered list of physical block IDs for this session.
    blocks: Vec<BlockId>,
    /// Number of tokens stored (max over all layers).
    seq_len: usize,
    /// Per-layer token counts so writing all layers for one token only advances
    /// the public sequence length once.
    layer_seq_lens: Vec<usize>,
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            seq_len: 0,
            layer_seq_lens: Vec::new(),
        }
    }

    pub fn push(&mut self, block: BlockId, tokens: usize) {
        self.blocks.push(block);
        self.seq_len += tokens;
    }

    /// Map a logical position to (block_index_in_table, offset_in_block).
    pub fn map(&self, logical_pos: usize) -> Option<(usize, usize)> {
        if logical_pos >= self.seq_len {
            return None;
        }
        let mut remaining = logical_pos;
        for (i, _) in self.blocks.iter().enumerate() {
            // Each block holds BLOCK_SIZE tokens
            if remaining < BLOCK_SIZE {
                return Some((i, remaining));
            }
            remaining -= BLOCK_SIZE;
        }
        None
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn block_ids(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.0).collect()
    }

    pub fn layer_seq_len(&self, layer: usize) -> usize {
        self.layer_seq_lens.get(layer).copied().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

/// Number of tokens per KV block. Must match attention kernel expectations.
pub const BLOCK_SIZE: usize = 16;

/// Paged KV cache — a pool of physical blocks managed by a free list.
pub struct PagedKvCache {
    /// [num_layers][num_blocks][block_size × dim] — K values.
    k: Vec<Vec<f32>>,
    /// [num_layers][num_blocks][block_size × dim] — V values.
    v: Vec<Vec<f32>>,
    dim: usize,
    num_layers: usize,
    total_blocks: usize,
    free_blocks: usize,
    /// Free block stack (index = available BlockId).
    free_list: Vec<BlockId>,
}

impl PagedKvCache {
    /// Create a paged KV cache with a fixed number of physical blocks.
    /// Total memory: num_layers × num_blocks × block_size × dim × 2 floats.
    pub fn new(num_layers: usize, dim: usize, num_blocks: usize) -> Self {
        let block_bytes = BLOCK_SIZE * dim;
        let k: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0f32; num_blocks * block_bytes])
            .collect();
        let v: Vec<Vec<f32>> = (0..num_layers)
            .map(|_| vec![0f32; num_blocks * block_bytes])
            .collect();
        let free_list: Vec<BlockId> = (0..num_blocks).map(BlockId).collect();
        Self {
            k,
            v,
            dim,
            num_layers,
            total_blocks: num_blocks,
            free_blocks: num_blocks,
            free_list,
        }
    }

    /// Allocate one physical block. Returns None if out of memory.
    pub fn alloc_block(&mut self) -> Option<BlockId> {
        self.free_list.pop().inspect(|_| {
            self.free_blocks -= 1;
        })
    }

    /// Free a physical block, returning it to the pool.
    pub fn free_block(&mut self, block: BlockId) {
        // Zero the block data
        let start = block.0 * BLOCK_SIZE * self.dim;
        let end = start + BLOCK_SIZE * self.dim;
        for layer in 0..self.num_layers {
            self.k[layer][start..end].fill(0.0);
            self.v[layer][start..end].fill(0.0);
        }
        self.free_list.push(block);
        self.free_blocks += 1;
    }

    /// Write K/V for a single layer at a given block + offset.
    pub fn write(
        &mut self,
        layer: usize,
        block: BlockId,
        offset_in_block: usize,
        k: &[f32],
        v: &[f32],
    ) {
        assert!(layer < self.num_layers);
        assert!(k.len() == self.dim && v.len() == self.dim);
        let block_start = block.0 * BLOCK_SIZE * self.dim;
        let pos = block_start + offset_in_block * self.dim;
        self.k[layer][pos..pos + self.dim].copy_from_slice(k);
        self.v[layer][pos..pos + self.dim].copy_from_slice(v);
    }

    /// Read K cache for a specific layer and block.
    pub fn read_k_block(&self, layer: usize, block: BlockId) -> &[f32] {
        let start = block.0 * BLOCK_SIZE * self.dim;
        let end = start + BLOCK_SIZE * self.dim;
        &self.k[layer][start..end]
    }

    /// Read V cache for a specific layer and block.
    pub fn read_v_block(&self, layer: usize, block: BlockId) -> &[f32] {
        let start = block.0 * BLOCK_SIZE * self.dim;
        let end = start + BLOCK_SIZE * self.dim;
        &self.v[layer][start..end]
    }

    pub fn free_blocks(&self) -> usize {
        self.free_blocks
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Append one token's K/V for a session, allocating new blocks as needed.
    /// Returns the block table entries after the append.
    pub fn session_append(
        &mut self,
        table: &mut BlockTable,
        layer: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        if table.layer_seq_lens.len() <= layer {
            table.layer_seq_lens.resize(layer + 1, 0);
        }
        let logical_pos = table.layer_seq_lens[layer];
        let block_idx = logical_pos / BLOCK_SIZE;
        let offset = logical_pos % BLOCK_SIZE;

        // Allocate a new block if needed
        if block_idx >= table.num_blocks() {
            let new_block = self
                .alloc_block()
                .ok_or_else(|| Error::Internal("KV block OOM".into()))?;
            table.push(new_block, 0); // tokens will be counted by seq_len update
        }

        let physical_block = table.blocks()[block_idx];
        self.write(layer, physical_block, offset, k, v);

        table.layer_seq_lens[layer] += 1;
        table.seq_len = table.seq_len.max(table.layer_seq_lens[layer]);

        Ok(())
    }
}

struct PagedSequenceSlot {
    table: BlockTable,
    k_linear: Vec<Vec<f32>>,
    v_linear: Vec<Vec<f32>>,
}

impl PagedSequenceSlot {
    fn new(num_layers: usize) -> Self {
        Self {
            table: BlockTable::new(),
            k_linear: (0..num_layers).map(|_| Vec::new()).collect(),
            v_linear: (0..num_layers).map(|_| Vec::new()).collect(),
        }
    }

    fn ensure_layer_len(&mut self, layer: usize, token_len: usize, dim: usize) {
        let float_len = token_len.saturating_mul(dim);
        if self.k_linear[layer].len() < float_len {
            self.k_linear[layer].resize(float_len, 0.0);
            self.v_linear[layer].resize(float_len, 0.0);
        }
    }
}

/// Sequence-oriented manager for `PagedKvCache`.
///
/// `PagedKvCache` is the physical vLLM-style block pool. This wrapper owns the
/// per-sequence block tables required by `SequenceKvCache`, so schedulers can
/// allocate/free resident sequences without knowing how pages are assigned.
///
/// The manager also keeps a compact host mirror per active sequence/layer. That
/// preserves the existing `KvLayerView<'_>` API for CPU/reference consumers while
/// exposing the block table for future paged-attention kernels.
pub struct PagedSequenceKvCache {
    cache: PagedKvCache,
    slots: Vec<Option<PagedSequenceSlot>>,
    max_blocks_per_sequence: usize,
}

impl PagedSequenceKvCache {
    pub fn new(num_layers: usize, dim: usize, num_blocks: usize, max_sequences: usize) -> Self {
        Self::with_max_blocks_per_sequence(num_layers, dim, num_blocks, max_sequences, num_blocks)
    }

    pub fn with_max_blocks_per_sequence(
        num_layers: usize,
        dim: usize,
        num_blocks: usize,
        max_sequences: usize,
        max_blocks_per_sequence: usize,
    ) -> Self {
        Self {
            cache: PagedKvCache::new(num_layers, dim, num_blocks),
            slots: (0..max_sequences).map(|_| None).collect(),
            max_blocks_per_sequence,
        }
    }

    pub fn cache(&self) -> &PagedKvCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut PagedKvCache {
        &mut self.cache
    }

    pub fn max_sequences(&self) -> usize {
        self.slots.len()
    }

    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn max_blocks_per_sequence(&self) -> usize {
        self.max_blocks_per_sequence
    }

    pub fn block_table(&self, handle: KvHandle) -> Result<&BlockTable> {
        self.slot(handle).map(|slot| &slot.table)
    }

    fn slot(&self, handle: KvHandle) -> Result<&PagedSequenceSlot> {
        self.slots
            .get(handle.0)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Internal(format!("inactive paged KV handle {}", handle.0)))
    }

    fn slot_mut(&mut self, handle: KvHandle) -> Result<&mut PagedSequenceSlot> {
        self.slots
            .get_mut(handle.0)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Internal(format!("inactive paged KV handle {}", handle.0)))
    }

    fn validate_write(&self, layer: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if layer >= self.cache.num_layers {
            return Err(Error::Internal(format!("KV layer {layer} out of range")));
        }
        if k.len() != self.cache.dim || v.len() != self.cache.dim {
            return Err(Error::Internal(format!(
                "KV dim mismatch: k={} v={} expected {}",
                k.len(),
                v.len(),
                self.cache.dim
            )));
        }
        Ok(())
    }
}

impl SequenceKvCache for PagedSequenceKvCache {
    fn storage_layout(&self) -> KvCacheLayout {
        KvCacheLayout::Paged {
            block_size: BLOCK_SIZE,
        }
    }

    fn layer_count(&self) -> usize {
        self.cache.num_layers
    }

    fn entry_dim(&self) -> usize {
        self.cache.dim
    }

    fn sequence_capacity(&self, handle: KvHandle) -> usize {
        if self.slots.get(handle.0).and_then(Option::as_ref).is_some() {
            self.max_blocks_per_sequence.saturating_mul(BLOCK_SIZE)
        } else {
            0
        }
    }

    fn alloc_sequence(&mut self) -> Result<KvHandle> {
        let Some(index) = self.slots.iter().position(Option::is_none) else {
            return Err(Error::Internal("no free paged KV sequence slots".into()));
        };
        self.slots[index] = Some(PagedSequenceSlot::new(self.cache.num_layers));
        Ok(KvHandle(index))
    }

    fn free_sequence(&mut self, handle: KvHandle) -> Result<()> {
        let slot = self
            .slots
            .get_mut(handle.0)
            .ok_or_else(|| Error::Internal(format!("unknown paged KV handle {}", handle.0)))?
            .take()
            .ok_or_else(|| Error::Internal(format!("inactive paged KV handle {}", handle.0)))?;

        for block in slot.table.blocks() {
            self.cache.free_block(*block);
        }
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
        self.validate_write(layer, k, v)?;
        let block_idx = pos / BLOCK_SIZE;
        let offset = pos % BLOCK_SIZE;
        if block_idx >= self.max_blocks_per_sequence {
            return Err(Error::Internal(format!(
                "paged KV position {pos} exceeds sequence capacity {}",
                self.max_blocks_per_sequence.saturating_mul(BLOCK_SIZE)
            )));
        }

        while block_idx >= self.slot(handle)?.table.num_blocks() {
            let block = self
                .cache
                .alloc_block()
                .ok_or_else(|| Error::Internal("KV block OOM".into()))?;
            self.slot_mut(handle)?.table.push(block, 0);
        }

        let block = self.slot(handle)?.table.blocks()[block_idx];
        self.cache.write(layer, block, offset, k, v);

        let dim = self.cache.dim;
        let token_len = pos + 1;
        let slot = self.slot_mut(handle)?;
        slot.ensure_layer_len(layer, token_len, dim);
        let start = pos * dim;
        let end = start + dim;
        slot.k_linear[layer][start..end].copy_from_slice(k);
        slot.v_linear[layer][start..end].copy_from_slice(v);

        if slot.table.layer_seq_lens.len() <= layer {
            slot.table.layer_seq_lens.resize(layer + 1, 0);
        }
        slot.table.layer_seq_lens[layer] = slot.table.layer_seq_lens[layer].max(token_len);
        slot.table.seq_len = slot.table.seq_len.max(slot.table.layer_seq_lens[layer]);
        Ok(())
    }

    fn layer_view(&self, handle: KvHandle, layer: usize) -> Result<KvLayerView<'_>> {
        if layer >= self.cache.num_layers {
            return Err(Error::Internal(format!("KV layer {layer} out of range")));
        }
        let slot = self.slot(handle)?;
        let seq_len = slot.table.layer_seq_len(layer);
        let float_len = seq_len.saturating_mul(self.cache.dim);
        let view = KvLayerView {
            k: &slot.k_linear[layer][..float_len],
            v: &slot.v_linear[layer][..float_len],
            seq_len,
            dim: self.cache.dim,
        };
        view.validate()?;
        Ok(view)
    }

    fn sequence_len(&self, handle: KvHandle) -> usize {
        self.slots
            .get(handle.0)
            .and_then(Option::as_ref)
            .map(|slot| slot.table.seq_len())
            .unwrap_or(0)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free_blocks() {
        let mut cache = PagedKvCache::new(2, 4, 10);
        assert_eq!(cache.free_blocks(), 10);

        let b0 = cache.alloc_block().unwrap();
        assert_eq!(cache.free_blocks(), 9);
        let _b1 = cache.alloc_block().unwrap();
        assert_eq!(cache.free_blocks(), 8);

        cache.free_block(b0);
        assert_eq!(cache.free_blocks(), 9);

        // Re-allocate should get b0 back (LIFO free list)
        let b2 = cache.alloc_block().unwrap();
        assert_eq!(b2, b0);
    }

    #[test]
    fn write_and_read_block() {
        let mut cache = PagedKvCache::new(1, 4, 4);
        let b = cache.alloc_block().unwrap();

        cache.write(0, b, 0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        cache.write(0, b, 1, &[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8]);

        let k_block = cache.read_k_block(0, b);
        assert_eq!(&k_block[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&k_block[4..8], &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn block_table_mapping() {
        let mut table = BlockTable::new();
        table.push(BlockId(5), 0); // placeholder
        table.push(BlockId(10), 0);

        // Manually set seq_len to simulate full blocks
        table.seq_len = 20; // 16 (first block) + 4 (second)

        assert_eq!(table.map(0), Some((0, 0)));
        assert_eq!(table.map(15), Some((0, 15)));
        assert_eq!(table.map(16), Some((1, 0)));
        assert_eq!(table.map(19), Some((1, 3)));
        assert!(table.map(20).is_none());
    }

    #[test]
    fn session_append_across_blocks() {
        let mut cache = PagedKvCache::new(1, 2, 4);
        let mut table = BlockTable::new();

        // Append BLOCK_SIZE + 1 tokens
        for i in 0..BLOCK_SIZE + 1 {
            cache
                .session_append(&mut table, 0, &[i as f32; 2], &[i as f32; 2])
                .unwrap();
        }

        assert_eq!(table.seq_len(), BLOCK_SIZE + 1);
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(cache.free_blocks(), 2); // 4 - 2 = 2 free
    }

    #[test]
    fn session_append_same_token_across_layers_keeps_len() {
        let mut cache = PagedKvCache::new(2, 2, 4);
        let mut table = BlockTable::new();

        cache
            .session_append(&mut table, 0, &[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        cache
            .session_append(&mut table, 1, &[5.0, 6.0], &[7.0, 8.0])
            .unwrap();

        assert_eq!(table.seq_len(), 1);
        assert_eq!(table.num_blocks(), 1);
        assert_eq!(&cache.read_k_block(0, table.blocks()[0])[..2], &[1.0, 2.0]);
        assert_eq!(&cache.read_k_block(1, table.blocks()[0])[..2], &[5.0, 6.0]);
    }

    #[test]
    fn oom_on_full_cache() {
        let mut cache = PagedKvCache::new(1, 2, 1);
        let mut table = BlockTable::new();
        // Fill the only block
        for _ in 0..BLOCK_SIZE {
            cache
                .session_append(&mut table, 0, &[1.0; 2], &[2.0; 2])
                .unwrap();
        }
        // Next token needs a new block → OOM
        assert!(cache
            .session_append(&mut table, 0, &[1.0; 2], &[2.0; 2])
            .is_err());
    }

    #[test]
    fn free_block_zeros_data() {
        let mut cache = PagedKvCache::new(1, 2, 2);
        let b = cache.alloc_block().unwrap();
        cache.write(0, b, 0, &[9.0, 9.0], &[9.0, 9.0]);

        cache.free_block(b);
        let k = cache.read_k_block(0, b);
        assert!(k.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn paged_sequence_api_alloc_append_view_and_free() {
        let mut cache = PagedSequenceKvCache::new(2, 2, 4, 2);
        assert_eq!(
            SequenceKvCache::storage_layout(&cache),
            KvCacheLayout::Paged {
                block_size: BLOCK_SIZE
            }
        );
        let handle = SequenceKvCache::alloc_sequence(&mut cache).unwrap();
        assert_eq!(handle, KvHandle(0));
        assert_eq!(cache.active_count(), 1);
        assert_eq!(
            SequenceKvCache::sequence_capacity(&cache, handle),
            4 * BLOCK_SIZE
        );

        SequenceKvCache::append_to_sequence(&mut cache, handle, 0, 0, &[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        SequenceKvCache::append_to_sequence(&mut cache, handle, 1, 0, &[10.0, 20.0], &[30.0, 40.0])
            .unwrap();

        let table = cache.block_table(handle).unwrap();
        assert_eq!(table.num_blocks(), 1);
        assert_eq!(table.seq_len(), 1);
        assert_eq!(table.layer_seq_len(0), 1);
        assert_eq!(table.layer_seq_len(1), 1);

        let view = SequenceKvCache::layer_view(&cache, handle, 1).unwrap();
        view.validate().unwrap();
        assert_eq!(view.seq_len, 1);
        assert_eq!(view.k, &[10.0, 20.0]);
        assert_eq!(view.v, &[30.0, 40.0]);

        SequenceKvCache::free_sequence(&mut cache, handle).unwrap();
        assert_eq!(cache.active_count(), 0);
        assert_eq!(cache.cache().free_blocks(), 4);
    }

    #[test]
    fn paged_sequence_api_spans_blocks() {
        let mut cache = PagedSequenceKvCache::new(1, 1, 3, 1);
        let handle = SequenceKvCache::alloc_sequence(&mut cache).unwrap();
        for pos in 0..=BLOCK_SIZE {
            SequenceKvCache::append_to_sequence(
                &mut cache,
                handle,
                0,
                pos,
                &[pos as f32],
                &[-(pos as f32)],
            )
            .unwrap();
        }

        let table = cache.block_table(handle).unwrap();
        assert_eq!(table.num_blocks(), 2);
        assert_eq!(table.seq_len(), BLOCK_SIZE + 1);
        assert_eq!(
            SequenceKvCache::sequence_len(&cache, handle),
            BLOCK_SIZE + 1
        );

        let view = SequenceKvCache::layer_view(&cache, handle, 0).unwrap();
        assert_eq!(view.k.len(), BLOCK_SIZE + 1);
        assert_eq!(view.k[0], 0.0);
        assert_eq!(view.k[BLOCK_SIZE], BLOCK_SIZE as f32);
        assert_eq!(view.v[BLOCK_SIZE], -(BLOCK_SIZE as f32));
    }

    #[test]
    fn paged_sequence_api_reports_block_oom() {
        let mut cache = PagedSequenceKvCache::new(1, 1, 1, 2);
        let first = SequenceKvCache::alloc_sequence(&mut cache).unwrap();
        let second = SequenceKvCache::alloc_sequence(&mut cache).unwrap();

        SequenceKvCache::append_to_sequence(&mut cache, first, 0, 0, &[1.0], &[1.0]).unwrap();
        let err = SequenceKvCache::append_to_sequence(&mut cache, second, 0, 0, &[2.0], &[2.0])
            .unwrap_err();
        assert!(format!("{err}").contains("KV block OOM"));
    }

    #[test]
    fn kv_cache_dtype_bytes_per_element() {
        assert_eq!(KvCacheDtype::Fp32.bytes_per_element(), 4);
        assert_eq!(KvCacheDtype::Fp16.bytes_per_element(), 2);
        assert_eq!(KvCacheDtype::Int8.bytes_per_element(), 1);
        assert_eq!(KvCacheDtype::Int4.bytes_per_element(), 1);
    }
}
