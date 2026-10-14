//! Host-only dedicated BF16 arena sizing; no allocation or initialization.
use ferrule_common::{Error, Result};

pub(crate) fn checked_grant(len: usize, budget: usize) -> Result<usize> {
    let invalid = || Error::Internal {
        message: "BF16 arena length/grant overflow or allocation budget exceeded".into(),
    };
    let bytes = len.checked_mul(size_of::<u16>()).ok_or_else(invalid)?;
    let grant = crate::cuda_allocation_grant_bytes(bytes).ok_or_else(invalid)?;
    if grant > budget {
        return Err(invalid());
    }
    Ok(grant)
}
