//! Allocation preflight using metadata only: these buffers must never be allocated.
use super::*;
#[test]
fn state_preflight_rejects_each_buffer_above_isize_max() {
    let base = GatedDeltaShape {
        key_heads: 1,
        value_heads: 1,
        key_dim: 1,
        value_dim: 1,
        kernel: 1,
    };
    let recurrent_dim = 1usize << (usize::BITS / 2 - 1);
    for shape in [
        GatedDeltaShape {
            kernel: isize::MAX as usize / 12 + 1,
            ..base
        },
        GatedDeltaShape {
            key_dim: recurrent_dim,
            value_dim: recurrent_dim / 2,
            ..base
        },
    ] {
        let (_, conv, recurrent) = shape.sizes().unwrap();
        assert!(conv.checked_mul(4).is_some() && recurrent.checked_mul(4).is_some());
        assert!(conv * 4 > isize::MAX as usize || recurrent * 4 > isize::MAX as usize);
        let schema = HybridStateSchema {
            layers: vec![HybridLayerSchema::GatedDeltaNet(shape)],
            entries: vec![],
            range: 0..1,
            total_layers: 1,
        };
        assert!(
            schema
                .per_sequence_state_bytes()
                .unwrap_err()
                .to_string()
                .contains("address space")
        );
        assert!(
            schema
                .create()
                .unwrap_err()
                .to_string()
                .contains("address space")
        );
    }
}
