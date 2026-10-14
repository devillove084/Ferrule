#[test]
fn owner_engine_has_no_send_or_leak_escape_hatch() {
    let source = include_str!("../src/engine/inference.rs");

    assert!(source.contains("pub trait InferenceEngine: 'static"));
    assert!(!source.contains("OwnerLocal"));
    assert!(!source.contains("unsafe impl"));
    assert!(!source.contains("mem::forget"));
}
