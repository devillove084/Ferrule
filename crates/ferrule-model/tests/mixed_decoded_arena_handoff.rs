//! CPU-only guard for the allocation-to-custody boundary; no CUDA context.
const MIXED: &str = include_str!("../src/transformer/standard/cuda/hybrid_rank/mixed.rs");

#[test]
fn dedicated_arena_handoff_precedes_any_post_allocation_observation() {
    let arena = MIXED
        .split_once("let arena_grants = cache.plan().arena_grants;")
        .unwrap()
        .1;
    let handoff = arena
        .split_once("let allocation =")
        .unwrap()
        .1
        .split_once("Err(error) => {")
        .unwrap()
        .0;
    let compact: String = handoff.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        compact
            .replace(",)", ")")
            .contains("allocate_bf16_arena_uninitialized(arena_grants/2,arena_grants)")
    );
    assert!(compact.contains("Ok(root)=>cache.install_arena(RefCell::new(root),arena_grants,"));
    for forbidden in [
        "allocator_metrics(",
        "counters(",
        "zero_bf16_buffer(",
        "sync_stream(",
    ] {
        assert!(
            !handoff.contains(forbidden),
            "unexpected work before retained root: {forbidden}"
        );
    }
    let installation = arena
        .split_once("let before = this.ops.allocator_metrics().granted_bytes;")
        .unwrap()
        .0;
    assert!(!installation.contains("zero_bf16_buffer("));
}

#[test]
fn slot_readiness_follows_full_decode_and_completion_classification() {
    let decode = MIXED
        .split_once("fn mixed_decoded_weight(")
        .unwrap()
        .1
        .split_once("fn mixed_view(")
        .unwrap()
        .0;
    let ready = decode.find(".mark_ready(descriptor)").unwrap();
    for boundary in [
        "cache.acquire(descriptor)?",
        ".decode_numeric_bf16_into(",
        ".decode_numeric_bf16_pair_into(",
        "self.mixed_numeric_completion(false, result)?",
    ] {
        assert!(decode.find(boundary).unwrap() < ready);
    }
}
