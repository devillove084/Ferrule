//! CPU preflight of the exact helpers used by opaque CUDA route entrypoints.
#[path = "../src/cuda/operators/bf16_route_metadata.rs"]
mod route_metadata;
use route_metadata::{PackedRoutes, Preparation, RouteBuffer, RouteOperation, validate_operands};
use std::{cell::Cell, sync::Arc};

#[test]
fn packed_routes_reject_counts_bounds_top8_and_nonfinite_weights() {
    for rows in [0, 9, usize::MAX] {
        assert!(PackedRoutes::new(&[0], &[1.0], rows).is_err());
    }
    for (ids, weights) in [
        (vec![], vec![]),
        (vec![0], vec![]),
        (vec![], vec![1.0]),
        (vec![-1], vec![1.0]),
        (vec![2], vec![1.0]),
        (vec![i32::MAX], vec![1.0]),
        (vec![0; 9], vec![1.0; 9]),
        (vec![0; 65], vec![1.0; 65]),
    ] {
        assert!(PackedRoutes::new(&ids, &weights, 2).is_err());
    }
    for weight in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fa00001),
    ] {
        assert!(PackedRoutes::new(&[0], &[weight], 1).is_err());
    }
    for rows in 1..=8 {
        let ids: Vec<_> = (0..8).flat_map(|_| (0..rows as i32).rev()).collect();
        let packed = PackedRoutes::new(&ids, &vec![1.0; ids.len()], rows).unwrap();
        assert_eq!(packed.routes, rows * 8);
        assert_eq!(packed.words.len(), 2 * ids.len());
    }
}

#[test]
fn packed_routes_preserve_order_f32_bits_and_exact_byte_offsets() {
    let bits = [
        0, 0x80000000, 1, 0x80000001, 0x3eaaaaab, 0x7f7fffff, 0xff7fffff,
    ];
    let weights = bits.map(f32::from_bits);
    let ids = [1, 0, 1, 1, 0, 1, 0];
    let packed = PackedRoutes::new(&ids, &weights, 2).unwrap();
    assert_eq!(packed.words[..7], ids.map(|row| row as u32));
    assert_eq!(packed.words[7..], bits);
    assert_eq!(packed.pointers(0x1000, 14).unwrap(), (0x1000, 0x101c));
    for (base, len) in [
        (0, 14),
        (0x1001, 14),
        (0x1002, 14),
        (0x1000, 13),
        (0x1000, 15),
        (u64::MAX - 3, 14),
    ] {
        assert!(packed.pointers(base, len).is_err());
    }
}

fn buffer(context: &Arc<usize>, start: u64, len: usize) -> RouteBuffer<'_, usize> {
    RouteBuffer {
        context,
        start,
        len,
    }
}

#[test]
fn route_preflight_exact_extents_owner_identity_aliases_and_overflow() {
    let owner = Arc::new(0usize);
    let foreign_same_ordinal = Arc::new(0usize);
    let packed = PackedRoutes::new(&[1, 0, 1], &[0.3, 0.5, 0.7], 2).unwrap();
    for operation in [RouteOperation::Gather, RouteOperation::Combine] {
        let (input_len, output_len) = match operation {
            RouteOperation::Gather => (10, 15),
            RouteOperation::Combine => (15, 10),
        };
        let validate = |metadata, input, output, width| {
            validate_operands(&owner, &packed, metadata, input, output, width, operation)
        };
        assert_eq!(
            validate(
                buffer(&owner, 0x1000, 6),
                buffer(&owner, 0x2000, input_len),
                buffer(&owner, 0x3000, output_len),
                5
            )
            .unwrap(),
            (0x1000, 0x100c)
        );
        for foreign in 0..3 {
            assert!(
                validate(
                    buffer(
                        if foreign == 0 {
                            &foreign_same_ordinal
                        } else {
                            &owner
                        },
                        0x1000,
                        6
                    ),
                    buffer(
                        if foreign == 1 {
                            &foreign_same_ordinal
                        } else {
                            &owner
                        },
                        0x2000,
                        input_len
                    ),
                    buffer(
                        if foreign == 2 {
                            &foreign_same_ordinal
                        } else {
                            &owner
                        },
                        0x3000,
                        output_len
                    ),
                    5,
                )
                .is_err()
            );
        }
        for delta in [-1isize, 1] {
            assert!(
                validate(
                    buffer(&owner, 0x1000, 6),
                    buffer(&owner, 0x2000, input_len.checked_add_signed(delta).unwrap()),
                    buffer(&owner, 0x3000, output_len),
                    5
                )
                .is_err()
            );
            assert!(
                validate(
                    buffer(&owner, 0x1000, 6),
                    buffer(&owner, 0x2000, input_len),
                    buffer(
                        &owner,
                        0x3000,
                        output_len.checked_add_signed(delta).unwrap()
                    ),
                    5
                )
                .is_err()
            );
        }
        for output_start in [
            0x1000,
            0x1002,
            0x100c,
            0x1016,
            0x2000,
            0x2002,
            0x2000 + (input_len * 2 - 2) as u64,
        ] {
            assert!(
                validate(
                    buffer(&owner, 0x1000, 6),
                    buffer(&owner, 0x2000, input_len),
                    buffer(&owner, output_start, output_len),
                    5
                )
                .is_err()
            );
        }
        for output_start in [0x1018, 0x2000 + (input_len * 2) as u64] {
            validate(
                buffer(&owner, 0x1000, 6),
                buffer(&owner, 0x2000, input_len),
                buffer(&owner, output_start, output_len),
                5,
            )
            .unwrap();
        }
        // Read/read aliases are permitted, but neither read may alias output.
        validate(
            buffer(&owner, 0x1000, 6),
            buffer(&owner, 0x1000, input_len),
            buffer(&owner, 0x3000, output_len),
            5,
        )
        .unwrap();
        for width in [0, usize::MAX, i32::MAX as usize] {
            assert!(
                validate(
                    buffer(&owner, 0x1000, 6),
                    buffer(&owner, 0x2000, input_len),
                    buffer(&owner, 0x3000, output_len),
                    width
                )
                .is_err()
            );
        }
        for start in [0, 0x2001, u64::MAX - 1] {
            assert!(
                validate(
                    buffer(&owner, 0x1000, 6),
                    buffer(&owner, start, input_len),
                    buffer(&owner, 0x3000, output_len),
                    5
                )
                .is_err()
            );
        }
    }
}

#[test]
fn preparation_never_publishes_after_h2d_fault_or_unknown_and_never_retries() {
    for fail_after_copy in [false, true] {
        let mut state = Preparation::Allocated;
        let copies = Cell::new(0);
        let result = state.submit(
            || {
                copies.set(copies.get() + 1);
                if fail_after_copy {
                    Ok(())
                } else {
                    Err(ferrule_common::Error::Internal {
                        message: "after H2D fault".into(),
                    })
                }
            },
            || {
                Err(ferrule_common::Error::Internal {
                    message: "completion Unknown".into(),
                })
            },
        );
        assert!(result.is_err());
        assert_eq!(copies.get(), 1);
        assert_eq!(state, Preparation::Unknown);
        assert!(state.require_ready().is_err());
        assert!(
            state
                .submit(|| panic!("must not recopy"), || panic!("must not fence"))
                .is_err()
        );
    }
    let mut state = Preparation::Allocated;
    assert!(state.require_ready().is_err());
    state.submit(|| Ok(()), || Ok(())).unwrap();
    state.require_ready().unwrap();
    assert!(
        state
            .submit(|| panic!("already prepared"), || panic!("no new fence"))
            .is_err()
    );
    assert_eq!(state, Preparation::Ready);
}
