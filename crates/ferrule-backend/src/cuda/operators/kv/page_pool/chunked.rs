//! Independently trimmable page/plane allocations. No logical ownership registry.
use super::*;
use crate::cuda::runtime::PinnedHostBuffer;

/// Installed before the first DMA; Unknown retains even a partially used frame.
pub(super) struct ChunkHostTransfer {
    planes: Vec<Vec<PinnedHostBuffer<f32>>>,
}

impl CudaKvPagePool {
    pub(in crate::cuda::operators::kv) fn new_chunked(
        planes: &[KvPlaneDescriptor],
        page_tokens: usize,
        max_slots: usize,
        staging_page_limit: usize,
    ) -> Result<Self> {
        let layout = PageLayout::new(planes, page_tokens, max_slots)?;
        if staging_page_limit == 0 || staging_page_limit > max_slots {
            return Err(pool_error(
                "staging page limit must be within physical capacity",
            ));
        }
        Ok(Self {
            planes: planes.to_vec().into_boxed_slice(),
            layout,
            plane_storage: Vec::new(),
            chunks: Some((0..max_slots).map(|_| Vec::new()).collect()),
            staging_page_limit,
            chunk_dma: None,
            metadata: MetadataState::with_slots(max_slots),
            pending: HashMap::new(),
            snapshots: HashMap::new(),
            next_token: 1,
            slot_epochs: vec![0; max_slots],
        })
    }

    pub(in crate::cuda::operators::kv) fn staging_page_limit(&self) -> Option<usize> {
        self.chunks.as_ref().map(|_| self.staging_page_limit)
    }

    pub(in crate::cuda::operators::kv) fn ensure_chunk(
        &mut self,
        context: &CudaOperators,
        slot: u32,
    ) -> Result<()> {
        let chunk = &mut self.chunks.as_mut().expect("chunked")[slot as usize];
        // Install each allocation before any initialization/copy can be submitted.
        while chunk.len() < self.planes.len() {
            let buffer = runtime::DeviceBuffer::uninitialized_dedicated(
                &context.stream_clone(),
                self.layout.page_elements[chunk.len()],
            )?;
            chunk.push(CudaF32Buffer::from_device_buffer(buffer));
        }
        Ok(())
    }

    pub(in crate::cuda::operators::kv) fn chunk_plane(
        &self,
        slot: u32,
        plane: usize,
    ) -> Result<&CudaF32Buffer> {
        self.chunks
            .as_ref()
            .and_then(|c| c.get(slot as usize))
            .and_then(|c| c.get(plane))
            .ok_or_else(|| pool_error("page allocation is absent or incomplete"))
    }

    pub(in crate::cuda::operators::kv) fn chunk_plane_mut(
        &mut self,
        slot: u32,
        plane: usize,
    ) -> Result<&mut CudaF32Buffer> {
        self.chunks
            .as_mut()
            .and_then(|c| c.get_mut(slot as usize))
            .and_then(|c| c.get_mut(plane))
            .ok_or_else(|| pool_error("page allocation is absent or incomplete"))
    }

    pub(in crate::cuda::operators::kv) fn release_chunked(&mut self, page: KvPageId) {
        if let Some(slot) = self.metadata.mappings.remove(&page) {
            self.recycle_transaction_slots([slot]);
        }
        self.snapshots.remove(&page);
    }

    pub(in crate::cuda::operators::kv) fn has_chunk_dma(&self) -> bool {
        self.chunk_dma.is_some()
    }

    pub(in crate::cuda::operators::kv) fn pending_chunk_dma_bytes(&self) -> usize {
        self.chunk_dma.as_ref().map_or(0, |dma| {
            dma.planes.iter().flatten().map(|p| p.len() * 4).sum()
        })
    }

    /// Called only after the transactional owner proves no original/staged/COW,
    /// committed-reader or prefix-DMA pins. Each synchronous D2H owns its buffer
    /// until completion; metadata publication follows the entire all-plane copy.
    pub(in crate::cuda::operators::kv) fn preempt_chunked(
        &mut self,
        context: &CudaOperators,
        pages: &[KvPageId],
    ) -> Result<()> {
        self.preempt_chunked_with_completion(context, pages, |context| {
            // The existing pinned-copy fault seam fails after submission. Keep
            // both operands in pool custody when no completion proof is minted.
            if context.failpoints().check_copy_event() {
                return Err(pool_error("injected chunked KV DMA completion failure"));
            }
            context.record_compute_event()?.synchronize()
        })
    }

    // Production and fault injection share the exact submission/custody path.
    pub(in crate::cuda::operators::kv) fn preempt_chunked_with_completion(
        &mut self,
        context: &CudaOperators,
        pages: &[KvPageId],
        complete: impl FnOnce(&CudaOperators) -> Result<()>,
    ) -> Result<()> {
        let stream = context.stream_clone();
        let mut transfer = ChunkHostTransfer { planes: Vec::new() };
        for _ in pages {
            transfer.planes.push(
                self.layout
                    .page_elements
                    .iter()
                    .map(|&len| {
                        PinnedHostBuffer::zeroed(stream.context(), len).map_err(Error::from)
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        self.chunk_dma = Some(transfer);
        for (index, &page) in pages.iter().enumerate() {
            let slot = self
                .physical_slot(page)
                .ok_or_else(|| pool_error("page not resident"))?;
            for plane in 0..self.planes.len() {
                let source = &self.chunks.as_ref().expect("chunked")[slot as usize][plane];
                let destination =
                    &mut self.chunk_dma.as_mut().expect("installed DMA").planes[index][plane];
                // The exact compute stream orders producer -> D2H. Both ends stay
                // owned in this pool through the final event, including errors.
                unsafe {
                    source
                        .as_device_buffer()
                        .copy_to_pinned_host_async(&stream, destination)?;
                }
            }
        }
        complete(context)?;
        let transfer = self.chunk_dma.take().expect("completed DMA");
        let snapshots = pages
            .iter()
            .zip(transfer.planes)
            .map(|(&page_id, planes)| KvHostSnapshot {
                page_id,
                planes: planes.iter().map(|p| p.as_slice().to_vec()).collect(),
            })
            .collect::<Vec<_>>();
        // No caller-supplied ACK, and no release after a partial or unknown copy.
        for snapshot in snapshots {
            let page = snapshot.page_id;
            self.snapshots
                .insert(page, StoredKvHostSnapshot::F32(snapshot));
            let slot = self
                .metadata
                .mappings
                .remove(&page)
                .expect("validated page");
            self.recycle_transaction_slots([slot]);
        }
        Ok(())
    }

    pub(in crate::cuda::operators::kv) fn restore_chunked(
        &mut self,
        context: &CudaOperators,
        pages: &[KvPageId],
    ) -> Result<()> {
        self.restore_chunked_with_completion(context, pages, |context| {
            // The existing pinned-copy fault seam fails after submission. Keep
            // both operands in pool custody when no completion proof is minted.
            if context.failpoints().check_copy_event() {
                return Err(pool_error("injected chunked KV DMA completion failure"));
            }
            context.record_compute_event()?.synchronize()
        })
    }

    pub(in crate::cuda::operators::kv) fn restore_chunked_with_completion(
        &mut self,
        context: &CudaOperators,
        pages: &[KvPageId],
        complete: impl FnOnce(&CudaOperators) -> Result<()>,
    ) -> Result<()> {
        let stream = context.stream_clone();
        let mut transfer = ChunkHostTransfer { planes: Vec::new() };
        for &page in pages {
            let snapshot = self
                .snapshot(page)
                .ok_or_else(|| pool_error("complete F32 snapshot required"))?;
            transfer.planes.push(
                snapshot
                    .planes
                    .iter()
                    .map(|values| {
                        PinnedHostBuffer::from_slice(stream.context(), values).map_err(Error::from)
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let slots = self.take_transaction_slots(pages.len())?;
        self.chunk_dma = Some(transfer);
        // Slots and both DMA operands remain charged on every submission error.
        for (index, &slot) in slots.iter().enumerate() {
            self.ensure_chunk(context, slot)?;
            for plane in 0..self.planes.len() {
                let source = &self.chunk_dma.as_ref().expect("installed DMA").planes[index][plane];
                let destination = &self.chunks.as_ref().expect("chunked")[slot as usize][plane];
                unsafe {
                    destination
                        .as_device_buffer()
                        .copy_from_pinned_host_async(&stream, source)?;
                }
            }
        }
        complete(context)?;
        self.chunk_dma.take();
        for (&page, slot) in pages.iter().zip(slots) {
            self.install_transaction_slot(page, slot);
            self.snapshots.remove(&page);
        }
        Ok(())
    }
}
