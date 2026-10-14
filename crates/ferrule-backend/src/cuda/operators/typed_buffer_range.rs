//! Host-only bounds for typed shared views; allocation custody stays in DeviceBuffer.
use ferrule_common::{Error, Result};

pub(crate) fn validate_slice<T>(
    base: u64,
    capacity: usize,
    offset: usize,
    len: usize,
) -> Result<()> {
    let invalid = || Error::Internal {
        message: "CUDA typed slice exceeds bounds or overflows element/byte/address range".into(),
    };
    let end = offset.checked_add(len).ok_or_else(invalid)?;
    if end > capacity {
        return Err(invalid());
    }
    let bytes = |count: usize| {
        count
            .checked_mul(size_of::<T>())
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(invalid)
    };
    let start = base.checked_add(bytes(offset)?).ok_or_else(invalid)?;
    start.checked_add(bytes(len)?).ok_or_else(invalid)?;
    Ok(())
}
