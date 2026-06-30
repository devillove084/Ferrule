//! Paged KV cache — block-based allocation with free list.
//!
//! Based on vLLM's PagedAttention: KV is stored in fixed-size blocks (pages).
//! Each session has a block table mapping logical positions → physical blocks.
//! Attention kernels read the block table to gather K/V.

use ferrule_core::Result;

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
                .ok_or_else(|| ferrule_core::Error::Internal("KV block OOM".into()))?;
            table.push(new_block, 0); // tokens will be counted by seq_len update
        }

        let physical_block = table.blocks()[block_idx];
        self.write(layer, physical_block, offset, k, v);

        table.layer_seq_lens[layer] += 1;
        table.seq_len = table.seq_len.max(table.layer_seq_lens[layer]);

        Ok(())
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
    fn kv_cache_dtype_bytes_per_element() {
        assert_eq!(KvCacheDtype::Fp32.bytes_per_element(), 4);
        assert_eq!(KvCacheDtype::Fp16.bytes_per_element(), 2);
        assert_eq!(KvCacheDtype::Int8.bytes_per_element(), 1);
        assert_eq!(KvCacheDtype::Int4.bytes_per_element(), 1);
    }
}
