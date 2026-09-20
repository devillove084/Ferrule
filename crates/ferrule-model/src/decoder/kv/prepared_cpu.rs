//! CPU opt-in to the shared physical prepared protocol.

use super::*;

impl PhysicalKvPreparedPool for CpuPagedKvPool {
    fn preflight_prepared(&self, transaction: &Self::Transaction) -> Result<()> {
        self.inner.preflight_commit(
            transaction
                .inner
                .as_ref()
                .ok_or_else(|| kv_error("CPU prepared physical handle absent"))?,
        )
    }

    fn install_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        _generation: u64,
    ) -> Result<KvEndProgress> {
        self.inner
            .commit_prepared(transaction.handle_mut()?)
            .map(map_end_progress)
    }

    fn poll_install_prepared(
        &mut self,
        _transaction: &mut Self::Transaction,
        _generation: u64,
    ) -> Result<KvEndProgress> {
        Err(kv_error("CPU install ACK is unknown; custody retained"))
    }

    fn abort_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        _generation: u64,
    ) -> Result<KvEndProgress> {
        self.inner
            .abort_prepared(transaction.handle_mut()?)
            .map(map_end_progress)
    }

    fn publish_prepared(&mut self, transaction: &Self::Transaction) -> Result<()> {
        self.inner.publish_prepared(
            transaction
                .inner
                .as_ref()
                .ok_or_else(|| kv_error("CPU prepared physical handle absent"))?,
        )
    }

    fn preflight_retirement(
        &self,
        transaction: &Self::Transaction,
        pages: &[KvPageId],
    ) -> Result<()> {
        self.inner.preflight_retirement(
            transaction
                .inner
                .as_ref()
                .ok_or_else(|| kv_error("CPU prepared physical handle absent"))?,
            pages,
        )
    }

    fn retire_prepared(
        &mut self,
        transaction: &mut Self::Transaction,
        _generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        self.inner
            .retire_prepared(transaction.handle_mut()?, pages)
            .map(map_end_progress)
    }

    fn finish_prepared(&mut self, mut transaction: Self::Transaction) -> Result<Vec<KvPageId>> {
        self.inner.finish_prepared(&mut transaction.inner)
    }
}
