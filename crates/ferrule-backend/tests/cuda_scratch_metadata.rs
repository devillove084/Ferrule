//! CPU-only regression tests for the metadata used by live CUDA preflight.
use ferrule_backend::cuda_allocation_grant_bytes as grant;
#[path = "../src/cuda/buffer_range.rs"]
mod buffer_range;
use buffer_range::F32Range;

#[test]
fn allocator_grant_class_boundaries_empty_and_overflow() {
    for (requested, expected) in [
        (0, 0),
        (1, 256),
        (255, 256),
        (256, 256),
        (257, 512),
        (512, 512),
        (513, 1024),
        (1_048_575, 1_048_576),
        (1_048_576, 1_048_576),
        (1_048_577, 2_097_152),
        (2_097_152, 2_097_152),
        (2_097_153, 4_194_304),
        (16_777_216, 16_777_216),
        (16_777_217, 18_874_368),
    ] {
        assert_eq!(grant(requested), Some(expected), "{requested}");
    }
    let last_page = usize::MAX & !2_097_151;
    assert_eq!(grant(last_page), Some(last_page));
    assert_eq!(grant(last_page + 1), None);
    assert_eq!(grant(usize::MAX), None);
}

#[test]
fn grant_matches_independent_oracle_across_classes() {
    for requested in (1usize..=4_194_305).step_by(127) {
        let expected = if requested <= 1_048_576 {
            let mut size = 256;
            while size < requested {
                size *= 2;
            }
            size
        } else {
            requested.div_ceil(2_097_152) * 2_097_152
        };
        assert_eq!(grant(requested), Some(expected));
    }
    let old = [10, 2, 2, 4, 10, 4, 4, 4].map(|n| grant(n * 4).unwrap());
    let new = [20, 4, 4, 8, 20, 8, 8, 8].map(|n| grant(n * 4).unwrap());
    assert_eq!(old.into_iter().chain(new).sum::<usize>(), 4096);
}

#[test]
fn checked_ranges_allow_only_explicit_exact_inplace_and_disjoint_writes() {
    let full = F32Range::new(4096, 16).unwrap();
    let prefix = F32Range::new(4096, 8).unwrap();
    let shifted = F32Range::new(4100, 16).unwrap();
    let adjacent = F32Range::new(4160, 8).unwrap();
    let empty = F32Range::new(4100, 0).unwrap();
    assert!(full.overlaps(prefix));
    for (read, write) in [
        (full, prefix),
        (prefix, full),
        (full, shifted),
        (shifted, full),
    ] {
        assert!(read.check_write(write, false).is_err());
        assert!(read.check_write(write, true).is_err());
    }
    assert!(full.check_write(full, false).is_err());
    assert!(full.check_write(full, true).is_ok());
    assert!(full.check_write(adjacent, false).is_ok());
    assert!(full.check_write(empty, false).is_ok());
    assert!(empty.check_write(full, false).is_ok());
    assert!(F32Range::new(u64::MAX - 3, 1).is_err());
    assert!(F32Range::new(0, usize::MAX).is_err());
    assert!(F32Range::new(u64::MAX, 0).is_ok());
}
