//! Driver-free checks used by the actual Entry and staging path.
use ferrule_common::{Error, Result};
use std::cell::Cell;
use std::collections::BTreeMap;
fn error(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

#[derive(Default)]
pub(super) struct StickyCompletion(Cell<bool>);
impl StickyCompletion {
    pub fn is_unknown(&self) -> bool {
        self.0.get()
    }
    pub fn cancel_unknown(&self) {
        self.0.set(true);
    }
    /// Private backend probe, never a public caller ACK. Err or unwind latches
    /// Unknown before touching a driver event, and subsequent success cannot heal it.
    pub fn probe<T>(&self, probe: impl FnOnce() -> Result<T>) -> Result<T> {
        if self.is_unknown() {
            return Err(error("CUDA completion is permanently unknown"));
        }
        self.0.set(true);
        let complete = probe()?;
        self.0.set(false);
        Ok(complete)
    }
}

pub(super) fn admit(pages: usize, limit: usize, active: usize) -> Result<()> {
    if limit == 0 || pages > limit || active != 0 {
        Err(error(
            "chunked KV requires one transaction within the staging page bound",
        ))
    } else {
        Ok(())
    }
}

pub(super) fn compact_slots(
    slots: impl IntoIterator<Item = u32>,
    limit: usize,
) -> Result<BTreeMap<u32, i32>> {
    let mut compact = BTreeMap::new();
    for slot in slots {
        let index =
            i32::try_from(compact.len()).map_err(|_| error("staging slots exceed i32 ABI"))?;
        if compact.len() >= limit || compact.insert(slot, index).is_some() {
            return Err(error("staging slots alias or exceed admission"));
        }
    }
    Ok(compact)
}
