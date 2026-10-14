//! Expected tensors are independent native HF captures, not a reimplemented HF forward.
use std::collections::BTreeMap;

fn fixture() -> BTreeMap<&'static str, Vec<f32>> {
    include_str!("support/hybrid_bf16_hf.txt")
        .lines()
        .map(|line| {
            let mut words = line.split_whitespace();
            (
                words.next().unwrap(),
                words.map(|v| v.parse().unwrap()).collect(),
            )
        })
        .collect()
}
fn round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounded = if v.is_nan() {
        (bits >> 16) | 0x40
    } else {
        bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
    };
    f32::from_bits(rounded << 16)
}

#[test]
fn native_gate_captures_require_sigmoid_and_product_boundaries() {
    let f = fixture();
    assert_eq!(f["scalar_expected"], [0.14453125]);
    let sigmoid = 1.0 / (1.0 + 0.9375f32.exp());
    assert_eq!(round(0.515625 * round(sigmoid)), 0.14453125);
    assert_eq!(round(0.515625 * sigmoid), 0.145_507_81);
    for (name, count, max_error) in [("attention", 38, 0.0078125), ("shared", 43, 0.00390625)] {
        let input = &f[format!("{name}_input").as_str()];
        let gates = &f[format!("{name}_gate").as_str()];
        let expected = &f[format!("{name}_expected").as_str()];
        let width = f[format!("{name}_shape").as_str()][1] as usize;
        let mut wrong_count = 0;
        let mut maximum = 0.0f32;
        for (i, &x) in input.iter().enumerate() {
            let gate = gates[if gates.len() == input.len() {
                i
            } else {
                i / width
            }];
            let s = 1.0 / (1.0 + (-gate).exp());
            assert_eq!(round(x * round(s)), expected[i], "{name} element {i}");
            let wrong = round(x * s);
            wrong_count += usize::from(wrong != expected[i]);
            maximum = maximum.max((wrong - expected[i]).abs());
        }
        assert_eq!(wrong_count, count);
        assert_eq!(maximum, max_error);
    }
}

#[test]
fn native_combine_capture_requires_f32_weights_and_each_bf16_add() {
    let f = fixture();
    let width = f["combine_shape"][1] as usize;
    let expected = &f["combine_expected"];
    let mut rounded = vec![0.0f32; expected.len()];
    let mut old = rounded.clone();
    let mut wrong_weights = rounded.clone();
    let mut wrong_add = rounded.clone();
    for (r, &row) in f["combine_rows"].iter().enumerate() {
        let weight = f["combine_weights"][r];
        for j in 0..width {
            let v = f["combine_input"][r * width + j];
            assert_eq!(v, round(v), "projection must already be BF16");
            let i = row as usize * width + j;
            rounded[i] = round(rounded[i] + round(v * weight));
            old[i] += v * weight;
            wrong_weights[i] = round(wrong_weights[i] + round(v * round(weight)));
            wrong_add[i] += round(v * weight);
        }
    }
    assert_eq!(&rounded, expected);
    assert_ne!(&wrong_weights, expected, "route weights must stay F32");
    assert_ne!(
        wrong_add.iter().copied().map(round).collect::<Vec<_>>(),
        *expected
    );
    let errors: Vec<_> = old
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (round(a) - b).abs())
        .collect();
    assert_eq!(errors.iter().filter(|&&e| e != 0.0).count(), 67);
    assert_eq!(errors.into_iter().fold(0.0f32, f32::max), 0.00390625);
    assert_eq!(expected[2], 0.43359375);
    assert_eq!(round(old[2]), 0.431_640_63);
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use ferrule_backend::cuda::operator_owner::{CudaBf16Buffer, CudaOperators};
    use ferrule_backend::cuda::operators::standard_bf16::Bf16GateLayout;

    fn upload(op: &CudaOperators, values: &[f32]) -> CudaBf16Buffer {
        op.upload_bf16_words(
            &values
                .iter()
                .map(|&v| (round(v).to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    fn one_launch(op: &CudaOperators) {
        let mut expected = op.counters();
        expected.kernel_launches = 0;
        expected.compute_kernel_launches = 0;
        assert_eq!(op.counters().kernel_launches, 1);
        assert_eq!(op.counters().compute_kernel_launches, 1);
        assert_eq!(expected, Default::default());
    }

    #[test]
    fn bf16_gate_layouts_reject_empty_overflow_and_native_index_overflow() {
        for (rows, width) in [(0, 1), (1, 0), (usize::MAX, 2), (2, i32::MAX as usize)] {
            for l in [
                Bf16GateLayout::Elementwise { rows, width },
                Bf16GateLayout::RowBroadcast { rows, width },
            ] {
                assert!(l.input_elements().is_err());
                assert!(l.gate_elements().is_err());
            }
        }
        assert_eq!(
            Bf16GateLayout::Elementwise { rows: 3, width: 5 }
                .gate_elements()
                .unwrap(),
            15
        );
        assert_eq!(
            Bf16GateLayout::RowBroadcast { rows: 3, width: 5 }
                .gate_elements()
                .unwrap(),
            3
        );
    }

    #[test]
    #[ignore = "requires sole-owner GPU assignment; exact native HF same-input tensors"]
    fn sigmoid_gate_matches_native_hf_elementwise_and_row_broadcast() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let f = fixture();
        for name in ["attention", "shared"] {
            let x = &f[format!("{name}_input").as_str()];
            let gates = &f[format!("{name}_gate").as_str()];
            let dims = &f[format!("{name}_shape").as_str()];
            let (rows, width) = (dims[0] as usize, dims[1] as usize);
            let input = upload(&op, x);
            let gate = upload(&op, gates);
            let mut output = upload(&op, &vec![99.0; x.len()]);
            let layout = if gates.len() == x.len() {
                Bf16GateLayout::Elementwise { rows, width }
            } else {
                Bf16GateLayout::RowBroadcast { rows, width }
            };
            op.reset_counters();
            op.sigmoid_gate_bf16_into(&input, &gate, &mut output, layout)
                .unwrap();
            one_launch(&op);
            assert_eq!(
                op.download_bf16_buffer(&output).unwrap(),
                f[format!("{name}_expected").as_str()]
            );
            assert_eq!(op.download_bf16_buffer(&input).unwrap(), *x);
            assert_eq!(op.download_bf16_buffer(&gate).unwrap(), *gates);
        }
        for layout in [
            Bf16GateLayout::Elementwise { rows: 3, width: 7 },
            Bf16GateLayout::RowBroadcast { rows: 3, width: 7 },
        ] {
            let input = upload(&op, &[0.515625; 21]);
            let gate = upload(&op, &vec![-0.9375; layout.gate_elements().unwrap()]);
            let mut output = op.zero_bf16_buffer(21).unwrap();
            op.sigmoid_gate_bf16_into(&input, &gate, &mut output, layout)
                .unwrap();
            assert_eq!(op.download_bf16_buffer(&output).unwrap(), [0.14453125; 21]);
        }
    }

    #[test]
    #[ignore = "requires sole-owner GPU assignment; full native HF expert result, no copied oracle"]
    fn weighted_combine_matches_native_hf_and_masks_routes() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let f = fixture();
        let width = f["combine_shape"][1] as usize;
        // Masked NaN routes must not be read, and an unrouted output row is zeroed.
        let mut rows: Vec<_> = f["combine_rows"].iter().map(|&v| v as i32).collect();
        rows.extend([-1, i32::MAX]);
        let mut values = f["combine_input"].clone();
        values.extend(vec![f32::NAN; 2 * width]);
        let mut weights = f["combine_weights"].clone();
        weights.extend([f32::NAN; 2]);
        let mut expected = f["combine_expected"].clone();
        expected.extend(vec![0.0; width]);
        let values = upload(&op, &values);
        let rows = op.upload_i32_buffer(&rows).unwrap();
        let weights = op.upload_f32_buffer(&weights).unwrap();
        let mut output = upload(&op, &vec![99.0; expected.len()]);
        op.reset_counters();
        op.weighted_combine_bf16_into(&values, &rows, &weights, &mut output, 9, width)
            .unwrap();
        one_launch(&op);
        assert_eq!(op.download_bf16_buffer(&output).unwrap(), expected);
        // No stateful accumulation from the previous result.
        op.weighted_combine_bf16_into(&values, &rows, &weights, &mut output, 9, width)
            .unwrap();
        assert_eq!(op.download_bf16_buffer(&output).unwrap(), expected);
    }

    #[test]
    #[ignore = "requires sole-owner GPU assignment; same-ordinal distinct owners, all operand preflight"]
    fn every_operand_owner_and_exact_extent_rejected_before_submission() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let foreign = CudaOperators::new_on_device(0).unwrap();
        // Test undersized, oversized and foreign buffers independently for each operand.
        for mode in 0..3 {
            for broadcast in [false, true] {
                for bad in 0..3 {
                    let sizes = [15, if broadcast { 3 } else { 15 }, 15];
                    let mut buffers: Vec<_> = sizes
                        .iter()
                        .enumerate()
                        .map(|(i, &n)| {
                            let owner = if mode == 2 && i == bad { &foreign } else { &op };
                            let n = if i == bad && mode < 2 {
                                if mode == 0 { n - 1 } else { n + 1 }
                            } else {
                                n
                            };
                            upload(owner, &vec![1.0; n])
                        })
                        .collect();
                    let mut output = buffers.pop().unwrap();
                    let layout = if broadcast {
                        Bf16GateLayout::RowBroadcast { rows: 3, width: 5 }
                    } else {
                        Bf16GateLayout::Elementwise { rows: 3, width: 5 }
                    };
                    op.reset_counters();
                    assert!(
                        op.sigmoid_gate_bf16_into(&buffers[0], &buffers[1], &mut output, layout)
                            .is_err()
                    );
                    assert_eq!(op.counters(), Default::default());
                    let owner = if mode == 2 && bad == 2 { &foreign } else { &op };
                    assert_eq!(
                        owner.download_bf16_buffer(&output).unwrap(),
                        vec![1.0; output.len()]
                    );
                }
            }
            for bad in 0..4 {
                let owner = |i| if mode == 2 && bad == i { &foreign } else { &op };
                let len = |i, n| {
                    if mode < 2 && bad == i {
                        if mode == 0 { n - 1 } else { n + 1 }
                    } else {
                        n
                    }
                };
                let values = upload(owner(0), &vec![1.0; len(0, 20)]);
                let rows = owner(1).upload_i32_buffer(&vec![0; len(1, 4)]).unwrap();
                let weights = owner(2).upload_f32_buffer(&vec![0.3; len(2, 4)]).unwrap();
                let mut output = upload(owner(3), &vec![1.0; len(3, 15)]);
                op.reset_counters();
                assert!(
                    op.weighted_combine_bf16_into(&values, &rows, &weights, &mut output, 3, 5)
                        .is_err()
                );
                assert_eq!(op.counters(), Default::default());
                assert_eq!(
                    owner(3).download_bf16_buffer(&output).unwrap(),
                    vec![1.0; output.len()]
                );
            }
        }
    }
}
