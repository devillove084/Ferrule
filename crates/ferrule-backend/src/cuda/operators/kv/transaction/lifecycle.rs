//! Page lifecycle operations share the same pins as transactions and views.
use super::*;

impl CudaPagedKvPool {
    pub fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        let pages: Vec<_> = unique(pages)?.into_iter().collect();
        let mut inner = self.inner.borrow_mut();
        inner.unpinned(&pages, None)?;
        for page in &pages {
            if inner.storage().physical_slot(*page).is_none()
                && !inner.storage().has_snapshot(*page)
            {
                return Err(error("CUDA release page is absent"));
            }
        }
        let context = inner.context.clone();
        for page in pages {
            inner.storage_mut().release(&context, page)?;
        }
        Ok(())
    }
    pub fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        let pages: Vec<_> = unique(pages)?.into_iter().collect();
        let mut inner = self.inner.borrow_mut();
        inner.unpinned(&pages, None)?;
        if pages
            .iter()
            .any(|p| inner.storage().physical_slot(*p).is_none())
        {
            return Err(error("CUDA preempt requires resident pages"));
        }
        let context = inner.context.clone();
        // Preempt publishes no metadata until all synchronous downloads succeed.
        if let Err(err) = inner.storage_mut().preempt(&context, &pages) {
            inner.poisoned = true;
            return Err(err);
        }
        Ok(())
    }
    pub fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        let pages: Vec<_> = unique(pages)?.into_iter().collect();
        let mut inner = self.inner.borrow_mut();
        inner.unpinned(&pages, None)?;
        if pages.len() > inner.storage().stats().free_slots
            || pages.iter().any(|p| !inner.storage().has_snapshot(*p))
        {
            return Err(error(
                "CUDA restore requires suspended pages and free capacity",
            ));
        }
        let context = inner.context.clone();
        // The legacy helper rolls back its reservation on I/O failure. Quarantine
        // the entire private storage before any clone can reuse those slots.
        if let Err(err) = inner.storage_mut().restore(&context, &pages) {
            inner.poisoned = true;
            return Err(err);
        }
        Ok(())
    }
    pub fn shutdown(&mut self) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        if inner.shutdown {
            return Ok(());
        }
        inner.available()?;
        if !inner.entries.is_empty() {
            return Err(error("CUDA shutdown cannot bypass active/prepared pins"));
        }
        if let Err(err) = inner
            .context
            .record_compute_event()
            .and_then(|fence| fence.synchronize())
        {
            inner.poisoned = true;
            return Err(err);
        }
        inner.shutdown = true;
        Ok(())
    }
}
