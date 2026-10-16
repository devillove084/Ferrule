//! One synchronous triple in the existing admitted metadata miss. No pinned
//! allocation or additional source bytes: aliases retain the original payloads.
use super::super::{CudaStandardDecoderOperators, cuda_error};
use crate::transformer::{ExpertAvailability, ExpertMetadata, ExpertProvider, PreparedSwiGlu};
use ferrule_common::Result;
use std::sync::Arc;

#[derive(Default)]
pub(in crate::transformer::standard::cuda) struct State {
    sources: Option<Arc<PreparedSwiGlu>>,
}
impl State {
    pub(in crate::transformer::standard::cuda) fn has_sources(&self) -> bool {
        self.sources.is_some()
    }
    fn begin(&mut self, expert: Arc<PreparedSwiGlu>) -> Result<()> {
        if self.has_sources() {
            return Err(cuda_error("grouped upload source custody unresolved"));
        }
        self.sources = Some(expert);
        Ok(())
    }
    fn finish(&mut self) {
        self.sources = None;
    }
}
impl Drop for State {
    fn drop(&mut self) {
        // A failed backend proof or unwind must not release the model's source
        // witnesses/host credit, even if later owner teardown fences succeed.
        std::mem::forget(self.sources.take());
    }
}

impl CudaStandardDecoderOperators {
    pub(in crate::transformer::standard::cuda) fn grouped_metadata_upload(
        &mut self,
        metadata: &ExpertMetadata,
        provider: &mut dyn ExpertProvider,
        rows: usize,
    ) -> Result<()> {
        let result = (|| {
            if !provider.supports_grouped_expert_upload() {
                return Err(cuda_error(
                    "grouped upload requires bounded provider source custody",
                ));
            }
            let key = self
                .bindings
                .expert_admission
                .clone()
                .ok_or_else(|| cuda_error("grouped upload requires exact cache admission"))?;
            let expert = match provider.expert(key.layer, key.expert)? {
                ExpertAvailability::Ready(expert) => expert,
                _ => {
                    return Err(cuda_error(
                        "metadata-ready grouped expert became unavailable",
                    ));
                }
            };
            metadata.validate_payload(&expert)?;
            let (actual, _, _) = self.expert_plan(&expert, rows)?;
            if actual != key {
                return Err(cuda_error("grouped payload belongs to another admission"));
            }
            let payloads = [expert.gate(), expert.up(), expert.down()].map(|p| {
                p.numeric_fp8()
                    .expect("numeric metadata validated")
                    .validated_payload()
                    .clone()
            });
            self.grouped_upload
                .as_mut()
                .expect("opt in")
                .begin(expert)?;
            let artifacts = self.ops.upload_validated_numeric_fp8_group_sync(payloads)?;
            self.bindings
                .publish_numeric_expert(&key, metadata, artifacts)?;
            self.grouped_upload.as_mut().expect("opt in").finish();
            Ok(())
        })();
        if result.is_err() {
            self.poisoned = true;
            self.cache_quarantine();
        }
        result
    }
}

#[cfg(test)]
#[path = "../../../../tests/grouped_numeric_upload/contracts.rs"]
mod tests;
