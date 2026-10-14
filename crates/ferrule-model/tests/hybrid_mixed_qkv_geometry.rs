//! Fixed Hello rank-local operands; operator diagnosis, never profile admission.
use ferrule_backend::cuda_plan::{
    NumericFp8Layout, NumericFp8ScaleType, bf16_gemmex::Bf16GemmExPlan,
};
use ferrule_model::transformer::hybrid_parallel::HybridMixedProjectionArithmetic;

fn layout(n: usize) -> NumericFp8Layout {
    NumericFp8Layout {
        n,
        k: 2048,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    }
}

#[test]
fn qkv_aligned_semantic_ranges_and_exact_concat_grant() {
    for ranges in [
        [0..1024, 2048..3072, 4096..6144],
        [1024..2048, 3072..4096, 6144..8192],
    ] {
        let mut packed_rows = 0;
        let mut scale_bytes = 0;
        for range in ranges {
            let source = NumericFp8Layout {
                row_origin: range.start,
                ..layout(range.len())
            };
            let packed = HybridMixedProjectionArithmetic::packed_gemmex_layout(source).unwrap();
            assert_eq!(
                packed.storage_lengths().unwrap(),
                source.storage_lengths().unwrap()
            );
            packed_rows += range.len();
            scale_bytes += packed.storage_lengths().unwrap().1;
        }
        assert_eq!(packed_rows, 4096);
        assert_eq!(scale_bytes, layout(4096).storage_lengths().unwrap().1);
    }
    let p = Bf16GemmExPlan::new(layout(4096), 1, usize::MAX, usize::MAX).unwrap();
    assert_eq!(p.staging_bytes(), 2 * ((1 + 4096) * 2048 + 4096));
    assert_eq!(p.cublas_workspace_bytes(), 32 << 20);
    assert_eq!(p.provider_bytes(), 16 << 20);
    assert!(
        Bf16GemmExPlan::new(
            layout(4096),
            1,
            p.storage_grant_bytes() - 1,
            p.provider_bytes()
        )
        .is_err()
    );
    assert!(
        Bf16GemmExPlan::new(
            layout(4096),
            1,
            p.storage_grant_bytes(),
            p.provider_bytes() - 1
        )
        .is_err()
    );
    Bf16GemmExPlan::new(layout(4096), 1, p.storage_grant_bytes(), p.provider_bytes()).unwrap();
    assert!(
        p.storage_grant_bytes()
            > Bf16GemmExPlan::new(layout(2048), 1, usize::MAX, usize::MAX)
                .unwrap()
                .storage_grant_bytes()
    );
}

#[test]
fn capital_m5_combined_qkv_plan_keeps_native_rows_and_full_width() {
    for rows in [1, 5, 8] {
        let p = Bf16GemmExPlan::new(layout(4096), rows, usize::MAX, usize::MAX).unwrap();
        assert_eq!(p.rows(), rows);
        assert_eq!((p.layout().n, p.layout().k), (4096, 2048));
        assert_eq!(
            p.staging_bytes(),
            (2 * ((rows + 4096) * 2048 + rows * 4096) + 255) & !255
        );
        assert!(
            Bf16GemmExPlan::new(
                layout(4096),
                rows,
                p.storage_grant_bytes() - 1,
                p.provider_bytes()
            )
            .is_err()
        );
    }
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use ferrule_backend::cuda::operator_owner::{CudaBf16Buffer, CudaOperators};
    use ferrule_backend::cuda::operators::recurrent_bf16::{
        Bf16CausalConv1dBuffers, Bf16GatedDeltaNetBuffers,
    };
    use ferrule_backend::cuda::operators::recurrent_bf16_chunk::CudaBf16ChunkWorkspace;
    use ferrule_backend::cuda::operators::{F32ToBf16RowsLayout, linear::Bf16GemmLayout};
    use ferrule_backend::cuda_plan::{Bf16ChunkPlan, CausalConv1dLayout, GatedDeltaNetLayout};
    use std::io::Write;
    use std::path::Path;

    fn words(path: &Path) -> Vec<u16> {
        let bytes = std::fs::read(path).unwrap();
        let (words, rest) = bytes.as_chunks::<2>();
        assert!(rest.is_empty());
        words.iter().map(|b| u16::from_le_bytes(*b)).collect()
    }
    fn save(path: &Path, values: &[f32]) {
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        for value in values {
            out.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    fn unchanged(actual: &[f32], expected: &[u16]) {
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|v| u32::from(*v) << 16)
                .collect::<Vec<_>>()
        );
    }
    fn pack(op: &CudaOperators, mut parts: Vec<CudaBf16Buffer>) -> CudaBf16Buffer {
        if parts.len() == 1 {
            return parts.pop().unwrap();
        }
        let mut packed = op.zero_f32_buffer(4096).unwrap();
        let mut offset = 0;
        for part in &parts {
            let expanded = op.bf16_to_f32(part).unwrap();
            op.copy_f32_range(&expanded, 0, &mut packed, offset, part.len())
                .unwrap();
            offset += part.len();
        }
        assert_eq!(offset, 4096);
        op.f32_rows_to_bf16_rne(&packed, F32ToBf16RowsLayout::packed(1, 1, 4096).unwrap())
            .unwrap()
    }

    #[test]
    #[ignore = "authenticated sole GPU owner; pinned fixed-input projection geometry only"]
    fn frozen_tp2_qkv_segmented_vs_concat() {
        assert_eq!(
            std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
            Ok("1")
        );
        assert!(std::env::var("FERRULE_PAPER_WORKER_TOKEN").is_ok());
        assert_eq!(
            std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
            Ok(":0:0")
        );
        let fixture = std::path::PathBuf::from(std::env::var_os("FERRULE_QKV_FIXTURE").unwrap());
        let out = std::path::PathBuf::from(std::env::var_os("FERRULE_QKV_OUTPUT").unwrap());
        std::fs::create_dir(&out).unwrap();
        let ops = [
            CudaOperators::new_on_device(0).unwrap(),
            CudaOperators::new_on_device(1).unwrap(),
        ];
        let mut reports = Vec::new();
        for (rank, op) in ops.iter().enumerate() {
            let read = |name: &str| words(&fixture.join(format!("rank{rank}.{name}.bf16")));
            let upload = |name: &str| op.upload_bf16_words(&read(name)).unwrap();
            let mut report = serde_json::json!({"rank":rank});
            {
                let raw = std::fs::read(fixture.join(format!("rank{rank}.weight.fp8"))).unwrap();
                let scales =
                    std::fs::read(fixture.join(format!("rank{rank}.scales.bf16"))).unwrap();
                let decoded = read("weight");
                assert_eq!(raw.len(), 4096 * 2048);
                assert_eq!(scales.len(), 32 * 16 * 2);
                assert_eq!(decoded.len(), raw.len());
                let x = upload("normalized_input");
                let a = upload("input.a");
                let b = upload("input.b");
                let al = upload("weight.a_log");
                let dt = upload("weight.dt_bias");
                let conv_w = upload("weight.conv");
                assert!(read("before.conv").iter().all(|v| *v == 0));
                assert!(
                    std::fs::read(fixture.join(format!("rank{rank}.before.ssm.f32")))
                        .unwrap()
                        .iter()
                        .all(|v| *v == 0)
                );
                let views = [1024, 2048, 4096]
                    .into_iter()
                    .flat_map(|n| {
                        [
                            Bf16GemmExPlan::new(layout(n), 1, usize::MAX, usize::MAX).unwrap(),
                            Bf16GemmExPlan::new_resident(n, 2048, 1, usize::MAX, usize::MAX)
                                .unwrap(),
                        ]
                    })
                    .collect::<Vec<_>>();
                let grant = views.iter().map(|p| p.storage_grant_bytes()).max().unwrap();
                let mut gemm = op
                    .prepare_bf16_gemmex_views(&views, grant, 16 << 20)
                    .unwrap();
                let chunk = Bf16ChunkPlan::new_prefill(
                    GatedDeltaNetLayout {
                        rows: 1,
                        key_heads: 8,
                        value_heads: 16,
                        key_dim: 128,
                        value_dim: 128,
                    },
                    32 << 20,
                )
                .unwrap();
                let mut scratch = op.zero_f32_buffer(chunk.workspace_elements()).unwrap();
                report["workspace_grant"] = grant.into();
                report["provider_version"] = gemm.provider_version().into();
                report["provider_observed_growth_bytes"] =
                    gemm.provider_observed_growth_bytes().into();
                for (name, resident, concat) in [
                    ("numeric_segmented", false, false),
                    ("resident_segmented", true, false),
                    ("numeric_concat", false, true),
                    ("resident_concat", true, true),
                ] {
                    let widths: &[usize] = if concat { &[4096] } else { &[1024, 1024, 2048] };
                    let mut parts = Vec::new();
                    let mut native_weights = Vec::new();
                    let mut numeric_weights = Vec::new();
                    let mut offset = 0;
                    for &n in widths {
                        let mut y = op.zero_bf16_buffer(n).unwrap();
                        if resident {
                            let weight = op
                                .upload_bf16_words(&decoded[offset * 2048..(offset + n) * 2048])
                                .unwrap();
                            op.reset_counters();
                            gemm.execute_bf16_view(
                                op,
                                Bf16GemmExPlan::new_resident(n, 2048, 1, grant, 16 << 20).unwrap(),
                                &weight,
                                &x,
                                &mut y,
                                Bf16GemmLayout::contiguous(1, n, 2048),
                            )
                            .unwrap();
                            native_weights.push(weight);
                        } else {
                            let weight = op
                                .upload_numeric_fp8_linear(
                                    layout(n),
                                    &raw[offset * 2048..(offset + n) * 2048],
                                    &scales[offset / 128 * 32..(offset + n) / 128 * 32],
                                )
                                .unwrap();
                            op.reset_counters();
                            gemm.execute_view(
                                op,
                                Bf16GemmExPlan::new(layout(n), 1, grant, 16 << 20).unwrap(),
                                &weight,
                                &x,
                                &mut y,
                            )
                            .unwrap();
                            numeric_weights.push(weight);
                        }
                        let c = op.counters();
                        assert_eq!(c.device_allocation_attempts, 0);
                        assert_eq!(c.host_to_device_copies, 0);
                        assert_eq!(c.device_to_host_copies, 0);
                        assert_eq!(c.artifact_uploads, 0);
                        assert_eq!(c.stream_wide_syncs, 0);
                        parts.push(y);
                        offset += n;
                    }
                    let qkv = pack(op, parts);
                    let mut history = op.zero_bf16_buffer(16384).unwrap();
                    let mut conv = op.zero_bf16_buffer(4096).unwrap();
                    let mut state = op.zero_f32_buffer(262144).unwrap();
                    let mut output = op.zero_bf16_buffer(2048).unwrap();
                    op.causal_depthwise_conv1d_silu_bf16_into(
                        Bf16CausalConv1dBuffers {
                            input: &qkv,
                            weight: &conv_w,
                            bias: None,
                            history: &mut history,
                            output: &mut conv,
                        },
                        CausalConv1dLayout {
                            rows: 1,
                            channels: 4096,
                            kernel_size: 4,
                        },
                    )
                    .unwrap();
                    op.gated_delta_net_bf16_chunked_into(
                        Bf16GatedDeltaNetBuffers {
                            qkv: &conv,
                            a: &a,
                            b: &b,
                            a_log: &al,
                            dt_bias: &dt,
                            state: &mut state,
                            output: &mut output,
                        },
                        Some(CudaBf16ChunkWorkspace::new(&mut scratch, chunk).unwrap()),
                        chunk,
                    )
                    .unwrap();
                    for (key, buffer) in [
                        ("qkv", &qkv),
                        ("conv", &conv),
                        ("history", &history),
                        ("output", &output),
                    ] {
                        let actual = op.download_bf16_buffer(buffer).unwrap();
                        if name == "numeric_segmented" {
                            let retained = match key {
                                "qkv" => "input.qkv",
                                "conv" => "convolved",
                                "history" => "conv_state",
                                _ => "recurrent_output",
                            };
                            unchanged(&actual, &read(retained));
                        }
                        save(&out.join(format!("rank{rank}.{name}.{key}.f32")), &actual);
                    }
                    let actual = op.download_f32_buffer(&state).unwrap();
                    if name == "numeric_segmented" {
                        assert_eq!(
                            actual
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                            std::fs::read(fixture.join(format!("rank{rank}.recurrent_state.f32")))
                                .unwrap()
                        );
                    }
                    save(&out.join(format!("rank{rank}.{name}.state.f32")), &actual);
                    op.sync_stream().unwrap();
                    assert!(!gemm.is_poisoned());
                }
                report["allocator_live_bytes_before_drop"] =
                    op.allocator_metrics().live_requested_bytes.into();
                op.sync_stream().unwrap();
            }
            op.sync_stream().unwrap();
            op.trim_device_allocator().unwrap();
            assert_eq!(op.allocator_metrics().live_requested_bytes, 0);
            report["allocator_live_bytes_after_drop"] = 0.into();
            reports.push(report);
        }
        std::fs::write(
            out.join("native.json"),
            serde_json::to_vec_pretty(&reports).unwrap(),
        )
        .unwrap();
    }
}
