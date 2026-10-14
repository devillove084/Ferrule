use ferrule_backend::cuda::operator_owner::{CudaBf16Buffer, CudaOperators};
use ferrule_backend::cuda_plan::F32RowsLayout;
use std::path::Path;

fn words(path: impl AsRef<Path>) -> Vec<u16> {
    let bytes = std::fs::read(path).unwrap();
    let (chunks, rest) = bytes.as_chunks::<2>();
    assert!(rest.is_empty());
    chunks.iter().copied().map(u16::from_le_bytes).collect()
}
fn resident(op: &CudaOperators, launches: u64) {
    let c = op.counters();
    assert_eq!(c.kernel_launches, launches);
    assert_eq!(c.device_allocation_attempts, 0);
    assert_eq!(c.host_to_device_copies, 0);
    assert_eq!(c.device_to_host_copies, 0);
    assert_eq!(c.stream_wide_syncs, 0);
}
#[test]
#[ignore = "sole GPU owner; frozen actual and independent HF ordinary norm suite"]
fn exact_frozen_norm_and_resident_resource_contract() {
    let dir = std::path::PathBuf::from(std::env::var_os("FERRULE_ORDINARY_NORM_FIXTURE").unwrap());
    let op = CudaOperators::new_on_device(0).unwrap();
    let cases = std::fs::read_to_string(dir.join("cases.tsv")).unwrap();
    let mut count = 0;
    for line in cases.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        let name = fields[0];
        let rows: usize = fields[1].parse().unwrap();
        let width: usize = fields[2].parse().unwrap();
        let epsilon = fields[3].parse().unwrap();
        let d = dir.join(name);
        let input_words = words(d.join("input.bf16"));
        let weight_words = words(d.join("weight.bf16"));
        let input = op.upload_bf16_words(&input_words).unwrap();
        let weight = op.upload_bf16_words(&weight_words).unwrap();
        let mut output = op.zero_bf16_buffer(rows * width).unwrap();
        op.reset_counters();
        op.offset_rms_norm_bf16_into(
            &input,
            &weight,
            &mut output,
            F32RowsLayout { rows, width },
            epsilon,
        )
        .unwrap();
        resident(&op, 1);
        let actual = op.download_bf16_buffer(&output).unwrap();
        let expected = words(d.join(if width > 4096 { "old.bf16" } else { "hf.bf16" }));
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &w)) in actual.iter().zip(&expected).enumerate() {
            let expected = f32::from_bits(u32::from(w) << 16);
            if name.starts_with("nonfinite") && expected.is_nan() {
                assert!(a.is_nan(), "{name} word{i}");
            } else {
                assert_eq!(a.to_bits(), expected.to_bits(), "{name} word{i}");
            }
        }
        assert_eq!(
            op.download_bf16_buffer(&input)
                .unwrap()
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
            input_words
        );
        assert_eq!(
            op.download_bf16_buffer(&weight)
                .unwrap()
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
            weight_words
        );
        count += 1;
    }
    assert_eq!(count, 105);
    eprintln!(
        "ordinary BF16 norm105 cases; exact finite words/nonfinite classification; one launch and zero alloc/copy/sync"
    );
}
#[test]
#[ignore = "sole GPU owner; every norm owner/extent/epsilon and writable alias preflight"]
fn owners_extents_epsilon_and_aliases_reject_before_submission() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let layout = F32RowsLayout { rows: 3, width: 5 };
    for wrong_owner in [false, true] {
        for bad in 0..3 {
            let mut buffers: Vec<CudaBf16Buffer> = [15, 5, 15]
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    let owner = if wrong_owner && i == bad {
                        &foreign
                    } else {
                        &op
                    };
                    owner
                        .upload_bf16_words(&vec![
                            0x3f80;
                            if !wrong_owner && i == bad { n - 1 } else { n }
                        ])
                        .unwrap()
                })
                .collect();
            let (reads, writes) = buffers.split_at_mut(2);
            op.reset_counters();
            assert!(
                op.offset_rms_norm_bf16_into(&reads[0], &reads[1], &mut writes[0], layout, 1e-6)
                    .is_err()
            );
            resident(&op, 0);
        }
    }
    let input = op.upload_bf16_words(&[0x3f80; 15]).unwrap();
    let weight = op.upload_bf16_words(&[0x3f80; 5]).unwrap();
    let mut output = op.zero_bf16_buffer(15).unwrap();
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        op.reset_counters();
        assert!(
            op.offset_rms_norm_bf16_into(&input, &weight, &mut output, layout, epsilon)
                .is_err()
        );
        resident(&op, 0);
    }
    for bad in [
        F32RowsLayout { rows: 0, ..layout },
        F32RowsLayout {
            width: usize::MAX,
            ..layout
        },
    ] {
        op.reset_counters();
        assert!(
            op.offset_rms_norm_bf16_into(&input, &weight, &mut output, bad, 1e-6)
                .is_err()
        );
        resident(&op, 0);
    }
    // Read/read aliases are legal; they do not weaken the writable exclusion.
    op.reset_counters();
    op.offset_rms_norm_bf16_into(
        &input,
        &input,
        &mut output,
        F32RowsLayout { rows: 1, width: 15 },
        1e-6,
    )
    .unwrap();
    resident(&op, 1);
    assert_eq!(op.download_bf16_buffer(&output).unwrap(), vec![2.0; 15]);
    assert_eq!(op.download_bf16_buffer(&input).unwrap(), vec![1.0; 15]);
}
