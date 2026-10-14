use super::*;

#[test]
fn streaming_gqa_abi_layout() {
    assert_eq!(size_of::<StreamingGqaArgs>(), 88);
    assert_eq!(align_of::<StreamingGqaArgs>(), 8);
    assert_eq!(std::mem::offset_of!(StreamingGqaArgs, query), 24);
    assert_eq!(std::mem::offset_of!(StreamingGqaArgs, stream), 80);
}
