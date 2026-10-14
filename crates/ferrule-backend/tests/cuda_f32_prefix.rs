#![cfg(feature = "cuda")]

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_backend::cuda::operators::recurrent::{F32RowsLayout, GateActivation};
use std::sync::Arc;

#[test]
#[ignore = "requires CUDA GPU; checked aliases preserve allocation and owner"]
fn prefix_bounds_owner_and_root_lifetime() {
    let ops = CudaOperators::new_on_device(0).unwrap();
    let root = ops.upload_f32_buffer(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    ops.reset_counters();
    assert!(root.prefix(5).is_err());
    assert!(root.prefix(usize::MAX).is_err());
    assert!(root.prefix(0).unwrap().is_empty());
    let full = root.prefix(4).unwrap();
    let prefix = root.prefix(2).unwrap();
    assert_eq!(prefix.len(), 2);
    assert!(
        prefix.prefix(3).is_err(),
        "nested views cannot regain capacity"
    );
    assert_eq!(
        prefix.as_device_buffer().cu_deviceptr(),
        root.as_device_buffer().cu_deviceptr()
    );
    assert!(Arc::ptr_eq(
        prefix.as_device_buffer().context(),
        root.as_device_buffer().context()
    ));
    assert_eq!(
        ops.counters(),
        Default::default(),
        "prefix does no GPU work"
    );
    drop(root);
    drop(full);
    let foreign = CudaOperators::new_on_device(0).unwrap();
    assert!(foreign.download_f32_buffer(&prefix).is_err());
    assert_eq!(ops.download_f32_buffer(&prefix).unwrap(), [1.0, 2.0]);
    drop(prefix);
    ops.sync_stream().unwrap();
}

#[test]
#[ignore = "requires CUDA GPU; exact logical length and alias rejection"]
fn prefix_satisfies_exact_guard_without_exposing_tail_or_bypassing_alias_guard() {
    let ops = CudaOperators::new_on_device(0).unwrap();
    let input = ops.upload_f32_buffer(&[1.0, 2.0]).unwrap();
    let gate = ops.upload_f32_buffer(&[0.0, 0.0]).unwrap();
    let mut root = ops.upload_f32_buffer(&[99.0; 4]).unwrap();
    let layout = F32RowsLayout { rows: 1, width: 2 };
    ops.reset_counters();
    assert!(
        ops.elementwise_gate_f32_into(&input, &gate, &mut root, layout, GateActivation::Sigmoid)
            .is_err()
    );
    let mut output = root.prefix(2).unwrap();
    assert!(
        ops.elementwise_gate_f32_into(
            &root.prefix(2).unwrap(),
            &gate,
            &mut output,
            layout,
            GateActivation::Sigmoid
        )
        .is_err()
    );
    assert_eq!(ops.counters(), Default::default());
    ops.elementwise_gate_f32_into(&input, &gate, &mut output, layout, GateActivation::Sigmoid)
        .unwrap();
    ops.sync_stream().unwrap();
    assert_eq!(
        ops.download_f32_buffer(&root).unwrap(),
        [0.5, 1.0, 99.0, 99.0]
    );
}
