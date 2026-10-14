//! Compile the production custody ledger without initializing or linking CUDA.
#[path = "../src/transformer/standard/cuda/root_submission.rs"]
mod transformer;

#[test]
fn opaque_operators_keep_the_real_barrier_and_scope_entry_is_not_an_operation() {
    let source = include_str!("../src/transformer/standard/cuda.rs");
    let synchronous = source
        .split("fn synchronous<T>")
        .nth(1)
        .unwrap()
        .split("// A caller may catch panics")
        .next()
        .unwrap();
    assert!(synchronous.contains("self.root_barrier(BarrierReason::OpaqueOperator)?;"));
    assert!(!synchronous.contains("self.root_submission.is_none()"));
    assert!(!synchronous.contains("begin_operation"));
    let source = include_str!("../src/transformer/standard/cuda/root_execution.rs");
    let entry = source
        .split("fn with_root_submission<T>")
        .nth(1)
        .unwrap()
        .split("fn poison_root_submission")
        .next()
        .unwrap();
    assert!(!entry.contains("begin_operation"));
    assert!(!entry.contains("finish_operation"));
    assert_eq!(
        source
            .matches("super::super::execute_standard_layer_inner(")
            .count(),
        1
    );
}

#[path = "unit/transformer/root_layer_integration.rs"]
mod root_layer_integration;
