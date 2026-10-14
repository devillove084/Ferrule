//! Read-only CUDA allocation sizing, also available without a CUDA runtime.

pub(crate) const MIN_ALIGNMENT: usize = 256;
pub(crate) const SMALL_LIMIT: usize = 1 << 20;
pub(crate) const ALLOCATOR_PAGE_BYTES: usize = 2 << 20;

/// Exact granted bytes for one ordinary CUDA allocator request (zero stays zero).
///
/// Uses the same size classes as live allocation: small blocks round up to a
/// power of two with a 256-byte minimum; larger blocks round to 2 MiB pages.
/// Returns `None` on overflow. No device, driver, allocation or payload access.
/// Excludes shared segment slack and retirement/quarantine of unrelated blocks.
pub fn cuda_allocation_grant_bytes(requested: usize) -> Option<usize> {
    if requested == 0 {
        Some(0)
    } else if requested <= SMALL_LIMIT {
        requested.max(MIN_ALIGNMENT).checked_next_power_of_two()
    } else {
        requested
            .checked_add(ALLOCATOR_PAGE_BYTES - 1)
            .map(|value| value & !(ALLOCATOR_PAGE_BYTES - 1))
    }
}
