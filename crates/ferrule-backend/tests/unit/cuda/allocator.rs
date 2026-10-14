use super::*;
#[test]
fn dedicated_kv_segment_never_lends_free_extent_to_weights_or_scratch() {
    let mut segment = Segment::new(1, 4096, 4096, SegmentKind::Dedicated);
    let offset = segment.allocate(256, 256).unwrap();
    let mut segments = vec![segment];
    for kind in [
        AllocationKind::Small,
        AllocationKind::Large,
        AllocationKind::Huge,
    ] {
        assert!(find_allocation(&mut segments, kind, 256, 256).is_none());
    }
    segments[0].release(offset, 256);
    assert!(segments[0].is_empty());
    assert!(find_allocation(&mut segments, AllocationKind::Small, 256, 256).is_none());
}
