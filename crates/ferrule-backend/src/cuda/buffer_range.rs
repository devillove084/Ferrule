//! Checked half-open byte ranges shared by F32 copy and kernel preflight.
use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct F32Range {
    start: u64,
    end: u64,
}
impl F32Range {
    pub(crate) fn new(start: u64, elements: usize) -> Result<Self> {
        let bytes = elements
            .checked_mul(4)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("F32 byte range overflow"))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| invalid("F32 address range overflow"))?;
        Ok(Self { start, end })
    }

    pub(crate) fn overlaps(self, other: Self) -> bool {
        self.start != self.end
            && other.start != other.end
            && self.start < other.end
            && other.start < self.end
    }

    pub(crate) fn check_write(self, write: Self, exact_inplace: bool) -> Result<()> {
        if self.overlaps(write) && !(exact_inplace && self == write) {
            return Err(invalid("overlapping writable buffer ranges"));
        }
        Ok(())
    }
}
fn invalid(message: &str) -> Error {
    Error::Internal {
        message: message.into(),
    }
}
