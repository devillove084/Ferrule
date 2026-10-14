use super::*;
use crate::cuda::operators::linear::{Bf16GemmLayout, CudaBf16Buffer, CudaOperators};
use crate::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};

const N: usize = 256;
const K: usize = 2048;
fn plan(m: usize) -> Bf16GemmExPlan {
    Bf16GemmExPlan::new(
        NumericFp8Layout {
            n: 2 * N,
            k: K,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        },
        m,
        128 << 20,
        16 << 20,
    )
    .unwrap()
}
fn cold_args(m: usize) -> (NumericArgs, NumericArgs) {
    let p = plan(m);
    let a = NumericArgs {
        m: m as u32,
        n: N as u32,
        k: K as u32,
        padded_k: K as u32,
        tile_rows: N as u32,
        scale_cols: (K / 128) as u32,
        row_origin: 0,
        column_origin: 0,
        scale_bytes: 2,
        lda: K as u32,
        ldd: N as u32,
        reserved: 0,
        activation: 0x10000,
        weight: 0x100000,
        scales: 0x200000,
        output: 0x1000000,
        workspace: 0x10000000,
        workspace_bytes: p.storage_bytes() as u64,
        stream: 0,
    };
    let b = NumericArgs {
        weight: 0x300000,
        scales: 0x400000,
        ..a
    };
    (a, b)
}
fn resident_args(m: usize) -> ResidentPairArgs {
    let p = plan(m);
    ResidentPairArgs {
        gemm: ResidentArgs {
            m: m as u32,
            n: (2 * N) as u32,
            k: K as u32,
            reserved: 0,
            input: 0x10000,
            weight: 0x100000,
            output: 0x10000000 + ((m + 2 * N) * K * 2) as u64,
            stream: 0,
        },
        gate: 0x1000000,
        up: 0x1000000,
        workspace: 0x10000000,
        workspace_bytes: p.storage_bytes() as u64,
    }
}

#[test]
fn pair_swiglu_cold_preflight_without_device_calls() {
    for m in [1, 2, 5] {
        let (a, b) = cold_args(m);
        assert_eq!(unsafe { ferrule_bf16_gemmex_can_pair_swiglu(&a, &b) }, 0);
        // Same destination is intentional only for the fused single-output ABI.
        assert_ne!(unsafe { ferrule_bf16_gemmex_can_pair(&a, &b) }, 0);
        assert_ne!(
            unsafe { ferrule_bf16_gemmex_can_pair_swiglu(std::ptr::null(), &b) },
            0
        );
        assert_ne!(
            unsafe { ferrule_bf16_gemmex_can_pair_swiglu(&a, std::ptr::null()) },
            0
        );
        for case in 0..22 {
            let (mut a, mut b) = cold_args(m);
            match case {
                0 => b.output += 256,
                1 => {
                    a.output = a.activation;
                    b.output = a.output;
                }
                2 => {
                    a.output = b.weight;
                    b.output = a.output;
                }
                3 => {
                    a.output = b.scales;
                    b.output = a.output;
                }
                4 => {
                    a.output = a.workspace;
                    b.output = a.output;
                }
                5 => {
                    a.output = u64::MAX - 255;
                    b.output = a.output;
                }
                6 => {
                    a.output += 2;
                    b.output = a.output;
                }
                7 => a.workspace_bytes -= 1,
                8 => a.workspace_bytes = u64::MAX,
                9 => b.activation += 256,
                10 => b.stream = 1,
                11 => b.n /= 2,
                12 => b.k /= 2,
                13 => b.scale_bytes = 4,
                14 => b.row_origin = 128,
                15 => b.column_origin = 128,
                16 => b.reserved = 1,
                17 => b.workspace += 256,
                18 => a.m = 33,
                19 => a.weight = a.workspace + 32 * 1024 * 1024,
                20 => b.scales = a.workspace + 32 * 1024 * 1024,
                _ => a.activation = a.workspace + 32 * 1024 * 1024,
            }
            assert_ne!(
                unsafe { ferrule_bf16_gemmex_can_pair_swiglu(&a, &b) },
                0,
                "M{m} case {case}"
            );
        }
    }
}

#[test]
fn pair_swiglu_resident_preflight_without_device_calls() {
    assert_eq!(std::mem::size_of::<ResidentPairArgs>(), 80);
    assert_ne!(
        unsafe { ferrule_bf16_gemmex_resident_pair_swiglu_can(std::ptr::null()) },
        0
    );
    for m in [1, 2, 5] {
        let a = resident_args(m);
        assert_eq!(
            unsafe { ferrule_bf16_gemmex_resident_pair_swiglu_can(&a) },
            0
        );
        assert_ne!(unsafe { ferrule_bf16_gemmex_resident_pair_can(&a) }, 0);
        for case in 0..19 {
            let mut a = resident_args(m);
            match case {
                0 => a.gate = a.gemm.input,
                1 => a.gate = a.gemm.weight,
                2 => a.gate = a.workspace,
                3 => a.gate = u64::MAX - 1,
                4 => a.gate += 1,
                5 => a.workspace_bytes -= 1,
                6 => a.workspace_bytes = u64::MAX,
                7 => a.workspace += 2,
                8 => a.gemm.output += 256,
                9 => a.gemm.n -= 1,
                10 => a.gemm.k -= 1,
                11 => a.gemm.m = 33,
                12 => a.gemm.reserved = 1,
                13 => a.gemm.input = a.workspace,
                14 => a.gemm.weight = a.workspace,
                15 => a.gate = 0,
                16 => a.gemm.weight = u64::MAX - 1,
                17 => a.gate = a.workspace + 32 * 1024 * 1024,
                _ => a.gate = a.gemm.output,
            }
            a.up = a.gate;
            assert_ne!(
                unsafe { ferrule_bf16_gemmex_resident_pair_swiglu_can(&a) },
                0,
                "M{m} case {case}"
            );
        }
        let mut a = resident_args(m);
        a.up += 2;
        assert_ne!(
            unsafe { ferrule_bf16_gemmex_resident_pair_swiglu_can(&a) },
            0
        );
    }
}

fn words(op: &CudaOperators, b: &CudaBf16Buffer) -> Vec<u16> {
    b.as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap()
}
fn slice(b: &CudaBf16Buffer, start: usize, len: usize) -> CudaBf16Buffer {
    CudaBf16Buffer::from_device_buffer(b.as_device_buffer().slice(start, len).unwrap())
}
fn counters(op: &CudaOperators, launches: u64) {
    let c = op.counters();
    assert_eq!(c.kernel_launches, launches);
    assert_eq!(c.device_allocation_attempts, 0);
    assert_eq!(c.stream_wide_syncs, 0);
    assert_eq!(c.device_to_host_copies, 0);
    assert_eq!(c.host_to_device_copies, 0);
}
fn exact(actual: &[u16], expected: &[u16], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a, b, "{label} word {i}: {a:04x} != {b:04x}");
    }
}

fn rank_payload(rank: usize) -> (Vec<u8>, Vec<u8>) {
    let codes = [
        0x00, 0x80, 0x01, 0x81, 0x38, 0xb8, 0x7e, 0xfe, 0x2b, 0xab, 0x4d, 0xcd,
    ];
    let mut encoded: Vec<u8> = (0..2 * N * K)
        .map(|i| codes[(i * 7 + i / K * 3 + i / 128 + rank * 5) % codes.len()])
        .collect();
    let scale_words = [0x0001u16, 0x3b91, 0x3dcd, 0x3f80, 0x4180, 0x0080, 0x3eab];
    let scales: Vec<u8> = (0..2 * N * K / 16384)
        .flat_map(|i| scale_words[(i + rank) % scale_words.len()].to_le_bytes())
        .collect();
    // Guaranteed all-zero gate/up channels alongside the dense stress rows.
    encoded[..K].fill(0x00);
    encoded[N * K..N * K + K].fill(0x80);
    (encoded, scales)
}

#[test]
fn pair_swiglu_rank_payloads_are_admitted_without_device_calls() {
    let half = NumericFp8Layout {
        n: N,
        ..plan(1).layout()
    };
    for rank in 0..2 {
        let (encoded, scales) = rank_payload(rank);
        plan(1)
            .layout()
            .validate_payload(&encoded, &scales)
            .unwrap();
        half.validate_payload(&encoded[..N * K], &scales[..scales.len() / 2])
            .unwrap();
        half.validate_payload(&encoded[N * K..], &scales[scales.len() / 2..])
            .unwrap();
    }
}

#[test]
#[ignore = "sole GPU owner; CUDA12.4/sm86 original pair+SwiGLU word oracle, rank-local N256 K2048 M1/M2/M5"]
fn pair_swiglu_rankpair_m1_m2_m5_exact() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let plans = [plan(1), plan(2), plan(5)];
    let mut owner = op
        .prepare_bf16_gemmex_views(&plans, plans[2].storage_grant_bytes(), 16 << 20)
        .unwrap();
    // Two distinct rank-local pairs; no distributed/model orchestration here.
    for rank in 0..2 {
        let (encoded, scales) = rank_payload(rank);
        let half = NumericFp8Layout {
            n: N,
            ..plans[0].layout()
        };
        let gate = op
            .upload_numeric_fp8_linear(half, &encoded[..N * K], &scales[..scales.len() / 2])
            .unwrap();
        let up = op
            .upload_numeric_fp8_linear(half, &encoded[N * K..], &scales[scales.len() / 2..])
            .unwrap();
        drop((encoded, scales));
        let mut resident = op.zero_bf16_buffer(2 * N * K).unwrap();
        owner
            .decode_numeric_bf16_pair_into(&op, [&gate, &up], &mut resident)
            .unwrap();
        let mut retained = Vec::new();
        for p in plans {
            let m = p.rows();
            let x = op
                .upload_bf16_words(
                    &(0..m * K)
                        .map(|i| [0x3f80, 0xbf40, 0x3eab, 0x0000, 0x8000][(i + i / K + rank) % 5])
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut g = op.zero_bf16_buffer(m * N).unwrap();
            let mut u = op.zero_bf16_buffer(m * N).unwrap();
            owner
                .execute_pair(&op, p, [&gate, &up], &x, [&mut g, &mut u])
                .unwrap();
            let expected = words(&op, &op.swiglu_bf16(&g, &u).unwrap());
            let mut output = op.zero_bf16_buffer(m * N).unwrap();
            let live = op.allocator_metrics().live_requested_bytes;
            op.reset_counters();
            owner
                .execute_pair_swiglu_view_into(&op, p, [&gate, &up], &x, &mut output)
                .unwrap();
            counters(&op, 5);
            assert_eq!(op.allocator_metrics().live_requested_bytes, live);
            exact(
                &words(&op, &output),
                &expected,
                &format!("cold rank{rank} M{m}"),
            );
            let layout = Bf16GemmLayout::contiguous(m, 2 * N, K);
            owner
                .execute_bf16_pair_view(&op, p, &resident, &x, [&mut g, &mut u], layout)
                .unwrap();
            exact(
                &words(&op, &op.swiglu_bf16(&g, &u).unwrap()),
                &expected,
                "resident unfused",
            );
            op.reset_counters();
            owner
                .execute_bf16_pair_swiglu_view_into(&op, p, &resident, &x, &mut output, layout)
                .unwrap();
            counters(&op, 2);
            exact(
                &words(&op, &output),
                &expected,
                &format!("resident rank{rank} M{m}"),
            );
            retained.push((p, x, expected, output));
        }
        // Prior downloads prove encoded reads complete. Reuse after source retirement.
        drop((gate, up));
        for (p, x, expected, mut output) in retained {
            owner
                .execute_bf16_pair_swiglu_view_into(
                    &op,
                    p,
                    &resident,
                    &x,
                    &mut output,
                    Bf16GemmLayout::contiguous(p.rows(), 2 * N, K),
                )
                .unwrap();
            exact(&words(&op, &output), &expected, "retired encoded pair");
        }
    }
}

#[test]
#[ignore = "sole GPU owner; original pair+SwiGLU at saturation, signed zeros, subnormal/normal and rounding edges"]
fn pair_swiglu_bf16_edges_m1_m2_m5_exact() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let plans = [plan(1), plan(2), plan(5)];
    let mut owner = op
        .prepare_bf16_gemmex_views(&plans, plans[2].storage_grant_bytes(), 16 << 20)
        .unwrap();
    // Sparse one-hot input exposes BF16 values at the GEMM output without a
    // separately rounded split GEMM. Include exp overflow, SiLU saturation,
    // adjacent BF16s, subnormals, underflow and product overflow.
    let edge = [
        0x0000u16, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x8080, 0x3f7f, 0x3f80, 0x3f81, 0xbf7f,
        0xbf80, 0xbf81, 0xc2b0, 0xc2b2, 0x42b0, 0x42b2, 0xc300, 0x4300, 0x7f7f, 0xff7f, 0x3eab,
        0xbeab,
    ];
    let mut weights = vec![0u16; 2 * N * K];
    for row in 0..2 * N {
        for col in 0..5 {
            weights[row * K + col] = edge[(row + col * 7) % edge.len()];
        }
    }
    let resident = op.upload_bf16_words(&weights).unwrap();
    drop(weights);
    for p in plans {
        let m = p.rows();
        let mut input = vec![0u16; m * K];
        for row in 0..m {
            input[row * K + row] = 0x3f80;
        }
        let x = op.upload_bf16_words(&input).unwrap();
        let mut g = op.zero_bf16_buffer(m * N).unwrap();
        let mut u = op.zero_bf16_buffer(m * N).unwrap();
        let mut output = op.zero_bf16_buffer(m * N).unwrap();
        let layout = Bf16GemmLayout::contiguous(m, 2 * N, K);
        owner
            .execute_bf16_pair_view(&op, p, &resident, &x, [&mut g, &mut u], layout)
            .unwrap();
        let expected = words(&op, &op.swiglu_bf16(&g, &u).unwrap());
        assert!(expected.iter().any(|w| w & 0x7fff == 0));
        op.reset_counters();
        owner
            .execute_bf16_pair_swiglu_view_into(&op, p, &resident, &x, &mut output, layout)
            .unwrap();
        counters(&op, 2);
        exact(&words(&op, &output), &expected, &format!("BF16 edges M{m}"));
    }
}

#[test]
#[ignore = "sole GPU owner; fused cold/resident owner, extents, alias, budgets, capture and sticky poison validation"]
fn pair_swiglu_validation_and_poison() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1);
    let baseline = op.allocator_metrics().live_requested_bytes;
    assert!(
        op.prepare_bf16_gemmex_views(&[p], p.storage_grant_bytes() - 1, p.provider_bytes())
            .is_err()
    );
    assert!(
        op.prepare_bf16_gemmex_views(&[p], p.storage_grant_bytes(), p.provider_bytes() - 1)
            .is_err()
    );
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
    let mut owner = op
        .prepare_bf16_gemmex_views(&[p], p.storage_grant_bytes(), p.provider_bytes())
        .unwrap();
    assert_eq!(
        op.allocator_metrics().live_requested_bytes - baseline,
        p.storage_bytes()
    );
    let half = NumericFp8Layout { n: N, ..p.layout() };
    let w = op
        .upload_numeric_fp8_linear(
            half,
            &vec![0x38; N * K],
            &[0x80, 0x3f].repeat(N * K / 16384),
        )
        .unwrap();
    let resident = op.zero_bf16_buffer(2 * N * K).unwrap();
    let x = op.zero_bf16_buffer(K).unwrap();
    let mut output = op.zero_bf16_buffer(N).unwrap();
    let layout = Bf16GemmLayout::contiguous(1, 2 * N, K);
    macro_rules! reject {
        ($call:expr) => {{
            op.reset_counters();
            assert!($call.is_err());
            counters(&op, 0);
        }};
    }
    let mut short = op.zero_bf16_buffer(N - 1).unwrap();
    let mut long = op.zero_bf16_buffer(N + 1).unwrap();
    let mut alias_x = slice(&x, 0, N);
    let mut alias_w = slice(&resident, 0, N);
    for bad in [&mut short, &mut long, &mut alias_x, &mut alias_w] {
        reject!(owner.execute_bf16_pair_swiglu_view_into(&op, p, &resident, &x, bad, layout));
    }
    for bad in [&mut short, &mut long, &mut alias_x] {
        reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &w], &x, bad));
    }
    let short_x = slice(&x, 0, K - 1);
    let long_x = op.zero_bf16_buffer(K + 1).unwrap();
    for bad in [&short_x, &long_x] {
        reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &w], bad, &mut output));
        reject!(owner.execute_bf16_pair_swiglu_view_into(
            &op,
            p,
            &resident,
            bad,
            &mut output,
            layout
        ));
    }
    let short_w = slice(&resident, 0, 2 * N * K - 1);
    let long_w = op.zero_bf16_buffer(2 * N * K + 1).unwrap();
    for bad in [&short_w, &long_w] {
        reject!(owner.execute_bf16_pair_swiglu_view_into(&op, p, bad, &x, &mut output, layout));
    }
    reject!(owner.execute_bf16_pair_swiglu_view_into(
        &op,
        p,
        &resident,
        &x,
        &mut output,
        Bf16GemmLayout::contiguous(1, N, K)
    ));
    reject!(owner.execute_bf16_pair_swiglu_view_into(
        &op,
        plan(2),
        &resident,
        &x,
        &mut output,
        Bf16GemmLayout::contiguous(2, 2 * N, K)
    ));
    reject!(owner.execute_pair_swiglu_view_into(&op, plan(2), [&w, &w], &x, &mut output));
    let wrong_scale = op
        .upload_numeric_fp8_linear(
            NumericFp8Layout {
                scale_type: NumericFp8ScaleType::F32,
                ..half
            },
            &vec![0x38; N * K],
            &1.0f32.to_le_bytes().repeat(N * K / 16384),
        )
        .unwrap();
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &wrong_scale], &x, &mut output));
    reject!(owner.execute_pair_swiglu_view_into(
        &op,
        p,
        [&wrong_scale, &wrong_scale],
        &x,
        &mut output
    ));
    let foreign = CudaOperators::new_on_device(0).unwrap();
    reject!(owner.execute_pair_swiglu_view_into(&foreign, p, [&w, &w], &x, &mut output));
    reject!(owner.execute_bf16_pair_swiglu_view_into(
        &foreign,
        p,
        &resident,
        &x,
        &mut output,
        layout
    ));
    let foreign_x = foreign.zero_bf16_buffer(K).unwrap();
    let mut foreign_y = foreign.zero_bf16_buffer(N).unwrap();
    let foreign_resident = foreign.zero_bf16_buffer(2 * N * K).unwrap();
    let foreign_w = foreign
        .upload_numeric_fp8_linear(
            half,
            &vec![0x38; N * K],
            &[0x80, 0x3f].repeat(N * K / 16384),
        )
        .unwrap();
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &foreign_w], &x, &mut output));
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&foreign_w, &w], &x, &mut output));
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &w], &foreign_x, &mut output));
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &w], &x, &mut foreign_y));
    reject!(owner.execute_bf16_pair_swiglu_view_into(
        &op,
        p,
        &foreign_resident,
        &x,
        &mut output,
        layout
    ));
    reject!(owner.execute_bf16_pair_swiglu_view_into(
        &op,
        p,
        &resident,
        &foreign_x,
        &mut output,
        layout
    ));
    reject!(owner.execute_bf16_pair_swiglu_view_into(
        &op,
        p,
        &resident,
        &x,
        &mut foreign_y,
        layout
    ));
    let stream = op.stream_clone();
    stream.begin_capture().unwrap();
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &w], &x, &mut output));
    reject!(owner.execute_bf16_pair_swiglu_view_into(&op, p, &resident, &x, &mut output, layout));
    drop(stream.end_capture().unwrap());
    assert!(!owner.is_poisoned());
    // Fault simulation has no outstanding work; no poisoned storage is reused.
    owner.native.poisoned.set(true);
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &w], &x, &mut output));
    reject!(owner.execute_bf16_pair_swiglu_view_into(&op, p, &resident, &x, &mut output, layout));
    owner.native.poisoned.set(false);
    op.sync_stream().unwrap();
    // Exercise the actual common submission-error transition, not only the flag.
    assert!(
        owner
            .native
            .submit_view(&stream, || COMPLETION_UNKNOWN)
            .is_err()
    );
    assert!(owner.is_poisoned());
    assert!(stream.context().graph_completion_unknown());
    reject!(owner.execute_pair_swiglu_view_into(&op, p, [&w, &w], &x, &mut output));
    reject!(owner.execute_bf16_pair_swiglu_view_into(&op, p, &resident, &x, &mut output, layout));
    // Preserve caller custody on Unknown, even though this injected failure
    // submitted no work. Native owner Drop quarantines its own provider storage.
    std::mem::forget((w, resident, x, output));
}
