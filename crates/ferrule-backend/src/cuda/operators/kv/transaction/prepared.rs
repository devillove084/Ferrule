//! Prepared installation and quarantine ACK journal.
use super::*;

impl CudaPagedKvPool {
    /// Ready requires the exact event recorded after this incarnation's writes.
    pub fn preflight_prepared(&self, tx: &CudaPagedKvTransaction) -> Result<()> {
        let inner = self.inner.borrow();
        let entry = inner.validate_active(tx)?;
        if entry.batch.is_none() || !entry.quiescent()? {
            return Err(error("CUDA KV exact writes fence is not ready"));
        }
        Ok(())
    }
    pub fn commit_prepared(
        &mut self,
        tx: &mut CudaPagedKvTransaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.preflight_prepared(tx)?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner.entry(tx)?;
        entry.generation(generation)?;
        let mapping = entry.plan.staged.clone();
        let mut old_slots = Vec::new();
        for (page, slot) in mapping {
            if let Some(old) = inner.storage_mut().install_transaction_slot(page, slot) {
                old_slots.push(old);
            }
        }
        let entry = inner.entries.get_mut(&tx.id).expect("validated");
        entry.quarantine = old_slots;
        entry.phase = Phase::Installed;
        entry.generation = Some(generation);
        Ok(KvEndProgress::Complete)
    }
    pub fn poll_install_ack(
        &self,
        tx: &CudaPagedKvTransaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        let inner = self.inner.borrow();
        let entry = inner.entry(tx)?;
        entry.generation(generation)?;
        if entry.phase != Phase::Installed || entry.generation.is_none() || !entry.quiescent()? {
            return Err(error("CUDA install ACK unknown; custody retained"));
        }
        Ok(KvEndProgress::Complete)
    }
    pub fn publish_prepared(&mut self, tx: &CudaPagedKvTransaction) -> Result<()> {
        let mut inner = self.inner.borrow_mut();
        if inner.entry(tx)?.phase != Phase::Installed {
            return Err(error("CUDA publication requires installed lease"));
        }
        inner.entries.get_mut(&tx.id).expect("validated").phase = Phase::Published;
        Ok(())
    }
    pub fn abort_prepared(
        &mut self,
        tx: &mut CudaPagedKvTransaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        let mut inner = self.inner.borrow_mut();
        let entry = inner.entry(tx)?;
        entry.generation(generation)?;
        if entry.phase == Phase::Aborted {
            return Ok(KvEndProgress::Complete);
        }
        if !matches!(entry.phase, Phase::Active | Phase::Aborting) || entry.entered {
            return Err(error("cannot abort installed or entered CUDA KV"));
        }
        let quiescent = entry.quiescent();
        let entry = inner.entries.get_mut(&tx.id).expect("validated");
        // Bind the decision before returning Pending/unknown. A caller cannot
        // switch generation or bypass this lease through ordinary commit/abort.
        entry.phase = Phase::Aborting;
        entry.generation = Some(generation);
        if !quiescent? {
            return Ok(KvEndProgress::Pending);
        }
        entry.quarantine = entry.plan.staged.values().copied().collect();
        entry.phase = Phase::Aborted;
        Ok(KvEndProgress::Complete)
    }
    pub fn preflight_retirement(
        &self,
        tx: &CudaPagedKvTransaction,
        pages: &[KvPageId],
    ) -> Result<()> {
        let pages: Vec<_> = unique(pages)?.into_iter().collect();
        let inner = self.inner.borrow();
        let entry = inner.entry(tx)?;
        if entry.phase == Phase::Retired {
            return if entry.retired == pages {
                Ok(())
            } else {
                Err(error("CUDA retirement retry changed page set"))
            };
        }
        if entry.phase != Phase::Published {
            return Err(error("CUDA retirement precedes publication"));
        }
        inner.unpinned(&pages, Some(tx.id))?;
        for page in pages {
            if !entry.plan.original.contains_key(&page)
                || (inner.storage().physical_slot(page).is_none()
                    && !inner.storage().has_snapshot(page))
            {
                return Err(error(
                    "CUDA retirement page is absent or not pinned by this owner",
                ));
            }
        }
        Ok(())
    }
    pub fn retire_prepared(
        &mut self,
        tx: &mut CudaPagedKvTransaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        self.preflight_retirement(tx, pages)?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner.entry(tx)?;
        entry.generation(generation)?;
        if entry.phase == Phase::Retired {
            return Ok(KvEndProgress::Complete);
        }
        if !entry.quiescent()? {
            return Ok(KvEndProgress::Pending);
        }
        let pages: Vec<_> = unique(pages)?.into_iter().collect();
        let slots: Vec<_> = pages
            .iter()
            .filter_map(|p| inner.storage_mut().detach_transaction_page(*p))
            .collect();
        let entry = inner.entries.get_mut(&tx.id).expect("validated");
        entry.quarantine.extend(slots);
        entry.retired = pages;
        entry.phase = Phase::Retired;
        Ok(KvEndProgress::Complete)
    }
    pub fn finish_prepared(
        &mut self,
        tx: &mut Option<CudaPagedKvTransaction>,
    ) -> Result<Vec<KvPageId>> {
        let handle = tx
            .as_ref()
            .ok_or_else(|| error("CUDA finish handle absent"))?;
        let mut inner = self.inner.borrow_mut();
        let entry = inner.entry(handle)?;
        if !matches!(entry.phase, Phase::Retired | Phase::Aborted) || !entry.quiescent()? {
            return Err(error("CUDA finish precedes exact cleanup ACK"));
        }
        let entry = inner.entries.remove(&handle.id).expect("validated");
        inner
            .storage_mut()
            .recycle_transaction_slots(entry.quarantine);
        tx.take();
        Ok(entry.retired)
    }
}
