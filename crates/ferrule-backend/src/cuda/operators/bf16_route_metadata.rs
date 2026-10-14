//! Host-only packed route preflight and submission publication.
use ferrule_common::{Error, Result};
use std::sync::Arc;

fn invalid(message: &str) -> Error {
    Error::Internal {
        message: format!("BF16 route metadata: {message}"),
    }
}

pub(super) struct PackedRoutes {
    pub(super) rows: usize,
    pub(super) routes: usize,
    pub(super) words: Box<[u32]>,
}
impl PackedRoutes {
    pub(super) fn new(row_ids: &[i32], weights: &[f32], rows: usize) -> Result<Self> {
        if !(1..=8).contains(&rows)
            || row_ids.is_empty()
            || row_ids.len() != weights.len()
            || row_ids.len() > rows * 8
        {
            return Err(invalid(
                "requires rows<=8, top-k<=8, 1<=R<=64 and equal counts",
            ));
        }
        let mut counts = [0usize; 8];
        for (&row, &weight) in row_ids.iter().zip(weights) {
            let row = usize::try_from(row).map_err(|_| invalid("negative row id"))?;
            if row >= rows || !weight.is_finite() {
                return Err(invalid("row out of bounds or nonfinite F32 weight"));
            }
            counts[row] += 1;
            if counts[row] > 8 {
                return Err(invalid("more than eight routes for one row"));
            }
        }
        let routes = row_ids.len();
        let words = row_ids
            .iter()
            .map(|&row| row as u32)
            .chain(weights.iter().map(|weight| weight.to_bits()))
            .collect();
        Ok(Self {
            rows,
            routes,
            words,
        })
    }
    pub(super) fn pointers(&self, base: u64, device_words: usize) -> Result<(u64, u64)> {
        let words = self
            .routes
            .checked_mul(2)
            .ok_or_else(|| invalid("word overflow"))?;
        if device_words != words
            || self.words.len() != words
            || !base.is_multiple_of(4)
            || base == 0
        {
            return Err(invalid("packed extent or alignment mismatch"));
        }
        let offset = byte_len(self.routes, size_of::<u32>())?;
        let weights = base
            .checked_add(offset)
            .ok_or_else(|| invalid("weight address overflow"))?;
        weights
            .checked_add(offset)
            .ok_or_else(|| invalid("packed address overflow"))?;
        Ok((base, weights))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Preparation {
    #[default]
    Allocated,
    Unknown,
    Ready,
}
impl Preparation {
    pub(super) fn submit(
        &mut self,
        upload: impl FnOnce() -> Result<()>,
        check_known: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if *self != Self::Allocated {
            return Err(invalid("metadata preparation may only be attempted once"));
        }
        // Caller already owns custody. Do not publish after partial H2D or
        // Unknown, and never retry/rewrite an in-flight source.
        *self = Self::Unknown;
        upload()?;
        check_known()?;
        *self = Self::Ready;
        Ok(())
    }
    pub(super) fn require_ready(self) -> Result<()> {
        if self != Self::Ready {
            return Err(invalid("metadata is unprepared or completion Unknown"));
        }
        Ok(())
    }
}

pub(super) struct RouteBuffer<'a, C> {
    pub(super) context: &'a Arc<C>,
    pub(super) start: u64,
    pub(super) len: usize,
}
#[derive(Clone, Copy)]
pub(super) enum RouteOperation {
    Gather,
    Combine,
}
fn byte_len(elements: usize, size: usize) -> Result<u64> {
    elements
        .checked_mul(size)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| invalid("byte extent overflow"))
}
fn region(start: u64, elements: usize, size: usize) -> Result<(u64, u64)> {
    let end = start
        .checked_add(byte_len(elements, size)?)
        .ok_or_else(|| invalid("address overflow"))?;
    if start == 0 || !start.is_multiple_of(size as u64) {
        return Err(invalid("null or unaligned operand"));
    }
    Ok((start, end))
}
fn shape(rows: usize, width: usize) -> Result<usize> {
    rows.checked_mul(width)
        .filter(|&n| width > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| invalid("invalid shape/extent"))
}

// Real DeviceBuffer context Arcs, not ordinals/driver handles. Host-only to
// exercise same-ordinal/different-owner and byte alias checks without CUDA.
pub(super) fn validate_operands<C>(
    owner: &Arc<C>,
    packed: &PackedRoutes,
    metadata: RouteBuffer<'_, C>,
    input: RouteBuffer<'_, C>,
    output: RouteBuffer<'_, C>,
    width: usize,
    operation: RouteOperation,
) -> Result<(u64, u64)> {
    let (input_rows, output_rows) = match operation {
        RouteOperation::Gather => (packed.rows, packed.routes),
        RouteOperation::Combine => (packed.routes, packed.rows),
    };
    if !Arc::ptr_eq(owner, metadata.context)
        || !Arc::ptr_eq(owner, input.context)
        || !Arc::ptr_eq(owner, output.context)
        || input.len != shape(input_rows, width)?
        || output.len != shape(output_rows, width)?
    {
        return Err(invalid("context owner or exact typed extent mismatch"));
    }
    let pointers = packed.pointers(metadata.start, metadata.len)?;
    let metadata = region(metadata.start, metadata.len, size_of::<u32>())?;
    let input = region(input.start, input.len, size_of::<u16>())?;
    let output = region(output.start, output.len, size_of::<u16>())?;
    for read in [metadata, input] {
        if output.0 < read.1 && read.0 < output.1 {
            return Err(invalid("overlapping writable buffer ranges"));
        }
    }
    Ok(pointers)
}
