use super::*;
use crate::cuda::operators::linear::{F32ToBf16RowsLayout, NumericFp8Precision};
use crate::cuda_plan::NUMERIC_BF16_GEMV_SCRATCH_BYTES;
use std::{io::Write, path::PathBuf, process::Command};

fn plan() -> NumericFp8LinearPlan {
    NumericFp8LinearPlan::new(
        NumericFp8Layout {
            n: 2048,
            k: 4096,
            row_origin: 0,
            column_origin: 0,
            scale_type: crate::cuda_plan::NumericFp8ScaleType::Bf16,
        },
        1,
        8 << 20,
        NumericFp8Precision::Bf16M1GemvF32Accumulate,
    )
    .unwrap()
}

#[test]
#[ignore = "sole GPU owner; pinned independent layer2 HF BF16 out projection, no full40"]
fn fixed_input_layer2_attention_output_gemv() {
    let dir = PathBuf::from(
        std::env::var_os("FERRULE_BF16_GEMV_DIR").expect("pinned supplemental fixture"),
    );
    let report = std::env::var_os("FERRULE_BF16_GEMV_REPORT").expect("fresh durable JSONL report");
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report)
        .unwrap();
    let sha = Command::new("sha256sum")
        .arg(dir.join("SHA256SUMS"))
        .output()
        .unwrap();
    assert!(sha.status.success());
    assert_eq!(
        String::from_utf8(sha.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        "cfa1c5613c179334bcafeacf3dc19dacd9ec947711b29ac701d3a155da1ce63d"
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let bytes = |suffix: &str, n: usize| {
        let b = std::fs::read(dir.join(format!("l2_AttentionOutput.{suffix}"))).unwrap();
        assert_eq!(b.len(), n);
        b
    };
    let words = |b: Vec<u8>| {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>()
    };
    let p = plan();
    let op = CudaOperators::new_on_device(0).unwrap();
    assert!(
        op.prepare_numeric_fp8_provider(p, p.provider_bytes() - 1)
            .is_err()
    );
    assert!(op.numeric_fp8_linear_workspace(p).is_err());
    assert!(op.numeric_gemv.borrow().is_none());
    op.prepare_numeric_fp8_provider(p, p.provider_bytes())
        .unwrap();
    let identity = std::sync::Arc::clone(op.numeric_gemv.borrow().as_ref().unwrap());
    let weight = op
        .upload_numeric_fp8_linear(
            p.layout(),
            &bytes("weight.fp8", 2048 * 4096),
            &bytes("scale.bf16", 16 * 32 * 2),
        )
        .unwrap();
    let input = op
        .upload_bf16_words(&words(bytes("input.bf16", 4096 * 2)))
        .unwrap();
    let expected = words(bytes("hf.bf16", 2048 * 2));
    let mut out = op.zero_f32_buffer(2048).unwrap();
    let mut bf = op.zero_bf16_buffer(2048).unwrap();
    let mut ws = op.numeric_fp8_linear_workspace(p).unwrap();
    assert_eq!(ws.allocated_bytes(), NUMERIC_BF16_GEMV_SCRATCH_BYTES);
    assert!(ws.provider_observed_growth_bytes() <= p.provider_bytes() as u64);
    assert_eq!(
        ws.provider_version(),
        Some(120405),
        "this frozen CUDA12.4 gate pins its provider"
    );
    op.f32_rows_to_bf16_rne_into(
        &out,
        &mut bf,
        F32ToBf16RowsLayout::packed(1, 1, 2048).unwrap(),
    )
    .unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
    let before_free = op.memory_info().unwrap().0;
    let before = op.allocator_metrics();
    let mut total_unequal = 0;
    for repeat in 0..3 {
        op.reset_counters();
        op.prepare_numeric_fp8_provider(p, p.provider_bytes())
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &identity,
            op.numeric_gemv.borrow().as_ref().unwrap()
        ));
        op.numeric_fp8_bf16_linear_into(&weight, &input, &mut out, &mut ws, p, 4096, 2048)
            .unwrap();
        op.f32_rows_to_bf16_rne_into(
            &out,
            &mut bf,
            F32ToBf16RowsLayout::packed(1, 1, 2048).unwrap(),
        )
        .unwrap();
        let counts = op.counters();
        assert_eq!(counts.device_allocation_attempts, 0);
        assert_eq!(counts.stream_wide_syncs, 0);
        let actual = op.download_bf16_buffer(&bf).unwrap();
        let raw = op.download_f32_buffer(&out).unwrap();
        let mut unequal = 0;
        let mut max_abs = 0f32;
        let mut first = None;
        for (i, (&a, &word)) in actual.iter().zip(&expected).enumerate() {
            let e = f32::from_bits(u32::from(word) << 16);
            if a.to_bits() != e.to_bits() || !raw[i].is_finite() {
                unequal += 1;
                if first.is_none() {
                    first = Some(i);
                }
            }
            max_abs = max_abs.max((a - e).abs());
        }
        let free = op.memory_info().unwrap().0;
        writeln!(report, "{{\"event\":\"comparison\",\"repeat\":{repeat},\"elements\":2048,\"unequal_bf16\":{unequal},\"max_abs\":{max_abs},\"first_index\":{},\"provider_reserve_bytes\":{},\"provider_observed_growth_bytes\":{},\"scratch_bytes\":{},\"free_before\":{before_free},\"free_after\":{free}}}",
            first.map_or_else(|| "null".into(), |v| v.to_string()), p.provider_bytes(), ws.provider_observed_growth_bytes(), ws.allocated_bytes()).unwrap();
        report.sync_all().unwrap();
        total_unequal += unequal;
        assert!(
            free >= before_free,
            "unexpected provider growth after fixed-shape initialization"
        );
        assert_eq!(
            op.allocator_metrics().live_granted_bytes,
            before.live_granted_bytes
        );
        assert!(!ws.is_poisoned());
    }
    // Rebinding scratch must retain the same handle, not initialize cuBLAS again.
    drop(ws);
    let ws = op.numeric_fp8_linear_workspace(p).unwrap();
    assert!(std::sync::Arc::ptr_eq(ws.gemv.as_ref().unwrap(), &identity));
    drop(ws);
    drop(identity);
    drop(bf);
    drop(out);
    drop(input);
    drop(weight);
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_granted_bytes, 0);
    drop(op);
    writeln!(report, "{{\"event\":\"complete\",\"comparisons\":3,\"elements\":6144,\"unequal_bf16\":{total_unequal},\"owner_dropped\":true}}").unwrap();
    report.sync_all().unwrap();
    assert_eq!(
        total_unequal, 0,
        "no tolerance relaxation for the frozen GEMV gate"
    );
}
