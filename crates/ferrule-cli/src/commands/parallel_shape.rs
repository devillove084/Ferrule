//! Checked host-side linear geometry, independent of physical rank workers.

use anyhow::{Context, Result, ensure};
use std::mem::size_of;

pub(crate) fn bytes<T>(count: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .context("allocation byte size overflow")
}

pub(crate) fn elements(rows: usize, width: usize) -> Result<usize> {
    ensure!(
        rows > 0 && width > 0,
        "shape dimensions must be greater than zero"
    );
    let count = rows
        .checked_mul(width)
        .context("shape element count overflow")?;
    bytes::<f32>(count)?;
    Ok(count)
}
