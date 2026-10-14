use super::*;
use crate::cuda::operators::linear::{Bf16GemmLayout, CudaBf16Buffer, CudaOperators};
use crate::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};

fn plan(m: usize, n: usize, k: usize) -> Bf16GemmExPlan {
    Bf16GemmExPlan::new(
        NumericFp8Layout {
            n,
            k,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        },
        m,
        512 << 20,
        16 << 20,
    )
    .unwrap()
}
fn decode_args() -> NumericArgs {
    NumericArgs {
        m: 1,
        n: 256,
        k: 128,
        padded_k: 128,
        tile_rows: 256,
        scale_cols: 1,
        row_origin: 0,
        column_origin: 0,
        scale_bytes: 2,
        lda: 128,
        ldd: 256,
        reserved: 0,
        activation: 0,
        weight: 0x10000,
        scales: 0x20000,
        output: 0x30000,
        workspace: 0,
        workspace_bytes: 0,
        stream: 0,
    }
}
#[test]
fn mixed_decode_preflight_without_device_calls() {
    assert_eq!(unsafe { ferrule_bf16_gemmex_decode_can(&decode_args()) }, 0);
    assert_ne!(
        unsafe { ferrule_bf16_gemmex_decode_can(std::ptr::null()) },
        0
    );
    for case in 0..17 {
        let mut a = decode_args();
        match case {
            0 => a.n = 127,
            1 => a.k = 129,
            2 => a.scale_bytes = 4,
            3 => a.scale_cols = 2,
            4 => a.row_origin = 1,
            5 => a.column_origin = 1,
            6 => a.output = a.weight,
            7 => a.output = a.scales,
            8 => a.output = u64::MAX - 1,
            9 => a.output += 1,
            10 => a.scales += 1,
            11 => a.weight = 0,
            12 => a.padded_k += 8,
            13 => a.workspace = 0x100000,
            14 => a.m = 0,
            15 => a.reserved = 1,
            _ => a.tile_rows = 128,
        }
        assert_ne!(
            unsafe { ferrule_bf16_gemmex_decode_can(&a) },
            0,
            "case {case}"
        );
    }
}
#[test]
fn mixed_pair_decode_preflight_without_device_calls() {
    let pair = || {
        let a = decode_args();
        let mut b = decode_args();
        b.weight = 0x60000;
        b.scales = 0x80000;
        b.output = a.output + u64::from(a.n) * u64::from(a.k) * 2;
        (a, b)
    };
    let (a, b) = pair();
    assert_eq!(unsafe { ferrule_bf16_gemmex_decode_pair_can(&a, &b) }, 0);
    assert_ne!(
        unsafe { ferrule_bf16_gemmex_decode_pair_can(std::ptr::null(), &b) },
        0
    );
    assert_ne!(
        unsafe { ferrule_bf16_gemmex_decode_pair_can(&a, std::ptr::null()) },
        0
    );
    for case in 0..15 {
        let (mut a, mut b) = pair();
        match case {
            0 => b.output += 2,
            1 => b.output = a.output,
            2 => b.weight = a.output,
            3 => b.scales = a.output,
            4 => a.weight = b.output,
            5 => a.scales = b.output,
            6 => b.n = 128,
            7 => b.k = 256,
            8 => b.scale_bytes = 4,
            9 => b.row_origin = 1,
            10 => b.column_origin = 1,
            11 => b.stream = 1,
            12 => b.scale_cols = 2,
            13 => {
                a.output = u64::MAX - 131071;
                b.output = a.output + 65536;
            }
            _ => a.output = 0,
        }
        assert_ne!(
            unsafe { ferrule_bf16_gemmex_decode_pair_can(&a, &b) },
            0,
            "case {case}"
        );
    }
    // Sharing immutable encoded reads is legal; only writes must be disjoint.
    let (a, mut b) = pair();
    b.weight = a.weight;
    b.scales = a.scales;
    assert_eq!(unsafe { ferrule_bf16_gemmex_decode_pair_can(&a, &b) }, 0);
}

fn pair_args() -> ResidentPairArgs {
    let p = plan(5, 1024, 2048);
    ResidentPairArgs {
        gemm: ResidentArgs {
            m: 5,
            n: 1024,
            k: 2048,
            reserved: 0,
            input: 0x10000,
            weight: 0x100000,
            output: 0x10000000 + (5 + 1024) * 2048 * 2,
            stream: 0,
        },
        gate: 0x1000000,
        up: 0x2000000,
        workspace: 0x10000000,
        workspace_bytes: p.storage_bytes() as u64,
    }
}
#[test]
fn mixed_pair_preflight_without_device_calls() {
    assert_eq!(std::mem::size_of::<ResidentPairArgs>(), 80);
    assert_eq!(
        unsafe { ferrule_bf16_gemmex_resident_pair_can(&pair_args()) },
        0
    );
    assert_ne!(
        unsafe { ferrule_bf16_gemmex_resident_pair_can(std::ptr::null()) },
        0
    );
    for case in 0..15 {
        let mut a = pair_args();
        match case {
            0 => a.gate = a.up,
            1 => a.gate = a.gemm.input,
            2 => a.up = a.gemm.weight,
            3 => a.gate = a.workspace,
            4 => a.gemm.input = a.workspace,
            5 => a.gemm.weight = a.workspace,
            6 => a.workspace_bytes -= 1,
            7 => a.workspace_bytes = u64::MAX,
            8 => a.gemm.output += 256,
            9 => a.gemm.n = 1023,
            10 => a.gemm.k = 2047,
            11 => a.up = u64::MAX - 1,
            12 => a.gate += 1,
            13 => a.gemm.reserved = 1,
            _ => a.gemm.m = 33,
        }
        assert_ne!(
            unsafe { ferrule_bf16_gemmex_resident_pair_can(&a) },
            0,
            "case {case}"
        );
    }
}
fn counters(op: &CudaOperators, launches: u64) {
    let c = op.counters();
    assert_eq!(c.kernel_launches, launches);
    assert_eq!(c.device_allocation_attempts, 0);
    assert_eq!(c.stream_wide_syncs, 0);
    assert_eq!(c.device_to_host_copies, 0);
    assert_eq!(c.host_to_device_copies, 0);
}
fn bits(op: &CudaOperators, b: &CudaBf16Buffer) -> Vec<u16> {
    op.download_bf16_buffer(b)
        .unwrap()
        .into_iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect()
}
fn slice(b: &CudaBf16Buffer, start: usize, len: usize) -> CudaBf16Buffer {
    CudaBf16Buffer::from_device_buffer(b.as_device_buffer().slice(start, len).unwrap())
}
#[test]
#[ignore = "sole GPU owner; CUDA12.4/sm86 M1/M5 real geometry cold/native/resident bit identity"]
fn mixed_decode_resident_single_pair_m1_m5() {
    let op = CudaOperators::new_on_device(0).unwrap();
    // Full widths: shared gate/up 1024, dense gate/up 8192, and down 2048.
    for (n, k) in [(512, 2048), (4096, 2048), (1024, 512)] {
        let plans = [plan(1, 2 * n, k), plan(5, 2 * n, k)];
        let mut owner = op
            .prepare_bf16_gemmex_views(&plans, plans[1].storage_grant_bytes(), 16 << 20)
            .unwrap();
        let codes = [
            0x01, 0x17, 0x2b, 0x38, 0x4d, 0x6e, 0x7e, 0x81, 0x97, 0xab, 0xb8, 0xcd, 0xee, 0xfe,
        ];
        let encoded: Vec<u8> = (0..2 * n * k)
            .map(|i| codes[(i * 7 + i / k * 3 + i / 128) % codes.len()])
            .collect();
        let scale_words = [0x3b91u16, 0x3c83, 0x3dcd, 0x3eab, 0x3f01];
        let scales: Vec<u8> = (0..2 * n * k / 16384)
            .flat_map(|i| scale_words[i % scale_words.len()].to_le_bytes())
            .collect();
        let full = op
            .upload_numeric_fp8_linear(plans[0].layout(), &encoded, &scales)
            .unwrap();
        let half = plan(1, n, k).layout();
        let gate = op
            .upload_numeric_fp8_linear(half, &encoded[..n * k], &scales[..scales.len() / 2])
            .unwrap();
        let up = op
            .upload_numeric_fp8_linear(half, &encoded[n * k..], &scales[scales.len() / 2..])
            .unwrap();
        // The device artifacts, not the host upload payloads, own the decode.
        drop((encoded, scales));
        let mut resident = op.zero_bf16_buffer(2 * n * k).unwrap();
        let mut whole = op.zero_bf16_buffer(2 * n * k).unwrap();
        op.reset_counters();
        owner
            .decode_numeric_bf16_pair_into(&op, [&gate, &up], &mut resident)
            .unwrap();
        counters(&op, 2);
        op.reset_counters();
        owner
            .decode_numeric_bf16_into(&op, &full, &mut whole)
            .unwrap();
        counters(&op, 1);
        assert_eq!(bits(&op, &resident), bits(&op, &whole));
        let mut retired_cases = Vec::new();
        for p in plans {
            let m = p.rows();
            let layout = Bf16GemmLayout::contiguous(m, 2 * n, k);
            let inputs = [0x3f80, 0xbf40, 0x3eab, 0xbe91, 0x3fc0];
            let x = op
                .upload_bf16_words(
                    &(0..m * k)
                        .map(|i| inputs[(i + i / k) % inputs.len()])
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut cold = op.zero_bf16_buffer(m * 2 * n).unwrap();
            let mut warm = op.zero_bf16_buffer(m * 2 * n).unwrap();
            let mut a = op.zero_bf16_buffer(m * n).unwrap();
            let mut b = op.zero_bf16_buffer(m * n).unwrap();
            op.reset_counters();
            owner.execute_view(&op, p, &full, &x, &mut cold).unwrap();
            counters(&op, 3);
            let expected = bits(&op, &cold);
            // Original cold decoder's bytes are the independent native oracle.
            let decoded_bytes = owner
                .native
                .storage
                .as_ref()
                .unwrap()
                .slice(m * k * 2, 2 * n * k * 2)
                .unwrap()
                .to_host_vec(&op.stream_clone())
                .unwrap();
            let decoded: Vec<u16> = decoded_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes([v[0], v[1]]))
                .collect();
            assert_eq!(bits(&op, &resident), decoded);
            let native = op.upload_bf16_words(&decoded).unwrap();
            drop((decoded, decoded_bytes));
            for w in [&resident, &native] {
                op.reset_counters();
                owner
                    .execute_bf16_view(&op, p, w, &x, &mut warm, layout)
                    .unwrap();
                counters(&op, 1);
                assert_eq!(bits(&op, &warm), expected);
            }
            op.reset_counters();
            owner
                .execute_pair(&op, p, [&gate, &up], &x, [&mut a, &mut b])
                .unwrap();
            counters(&op, 5);
            let gate_expected = bits(&op, &a);
            let up_expected = bits(&op, &b);
            for row in 0..m {
                assert_eq!(
                    &gate_expected[row * n..(row + 1) * n],
                    &expected[row * 2 * n..row * 2 * n + n]
                );
                assert_eq!(
                    &up_expected[row * n..(row + 1) * n],
                    &expected[row * 2 * n + n..(row + 1) * 2 * n]
                );
            }
            for w in [&resident, &native] {
                op.reset_counters();
                owner
                    .execute_bf16_pair_view(&op, p, w, &x, [&mut a, &mut b], layout)
                    .unwrap();
                counters(&op, 2);
                assert_eq!(bits(&op, &a), gate_expected);
                assert_eq!(bits(&op, &b), up_expected);
            }
            retired_cases.push((p, x, gate_expected, up_expected));
        }
        // Downloads above prove all encoded reads complete before retirement.
        // The cache must remain sufficient for both warmed M1/M5 geometries.
        drop((full, gate, up, whole));
        for (p, x, gate_expected, up_expected) in retired_cases {
            let mut a = op.zero_bf16_buffer(p.rows() * n).unwrap();
            let mut b = op.zero_bf16_buffer(p.rows() * n).unwrap();
            op.reset_counters();
            owner
                .execute_bf16_pair_view(
                    &op,
                    p,
                    &resident,
                    &x,
                    [&mut a, &mut b],
                    Bf16GemmLayout::contiguous(p.rows(), 2 * n, k),
                )
                .unwrap();
            counters(&op, 2);
            assert_eq!(bits(&op, &a), gate_expected);
            assert_eq!(bits(&op, &b), up_expected);
        }
    }
}

#[test]
#[ignore = "sole GPU owner; resident decode/pair preflight owner, extent, strides, capture, Unknown"]
fn mixed_decode_pair_rejections() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 1024, 2048);
    let mut owner = op
        .prepare_bf16_gemmex_views(&[p], p.storage_grant_bytes(), 16 << 20)
        .unwrap();
    let l = plan(1, 512, 2048).layout();
    let w = op
        .upload_numeric_fp8_linear(
            l,
            &vec![0x38; l.n * l.k],
            &[0x80, 0x3f].repeat(l.n * l.k / 16384),
        )
        .unwrap();
    let mut resident = op.zero_bf16_buffer(1024 * 2048).unwrap();
    let mut dst = slice(&resident, 0, 512 * 2048);
    let mut short = slice(&resident, 0, 512 * 2048 - 1);
    let mut long = slice(&resident, 0, 512 * 2048 + 1);
    for bad in [&mut short, &mut long] {
        assert!(owner.decode_numeric_bf16_into(&op, &w, bad).is_err());
    }
    let wrong_scale = NumericFp8Layout {
        scale_type: NumericFp8ScaleType::F32,
        ..l
    };
    let f32w = op
        .upload_numeric_fp8_linear(
            wrong_scale,
            &vec![0x38; l.n * l.k],
            &1.0f32.to_le_bytes().repeat(l.n * l.k / 16384),
        )
        .unwrap();
    assert!(
        owner
            .decode_numeric_bf16_into(&op, &f32w, &mut dst)
            .is_err()
    );
    let mut oversized = op.zero_bf16_buffer(1024 * 2048 + 1).unwrap();
    let mut undersized = slice(&resident, 0, 1024 * 2048 - 1);
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&w, &w], &mut oversized)
            .is_err()
    );
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&w, &w], &mut undersized)
            .is_err()
    );
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&w, &f32w], &mut resident)
            .is_err()
    );
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&f32w, &f32w], &mut resident)
            .is_err()
    );
    let x = op.zero_bf16_buffer(2048).unwrap();
    let x_padded = op.zero_bf16_buffer(2049).unwrap();
    let mut a = op.zero_bf16_buffer(512).unwrap();
    let mut b = op.zero_bf16_buffer(512).unwrap();
    let layout = Bf16GemmLayout::contiguous(1, 1024, 2048);
    let wrong_layout = Bf16GemmLayout::contiguous(1, 1024, 2049);
    assert!(
        owner
            .execute_bf16_pair_view(&op, p, &resident, &x, [&mut a, &mut b], wrong_layout)
            .is_err()
    );
    assert!(
        owner
            .execute_bf16_pair_view(&op, p, &resident, &x_padded, [&mut a, &mut b], layout)
            .is_err()
    );
    assert!(
        owner
            .execute_bf16_pair_view(&op, p, &dst, &x, [&mut a, &mut b], layout)
            .is_err()
    );
    assert!(
        owner
            .execute_bf16_pair_view(
                &op,
                plan(5, 1024, 2048),
                &resident,
                &x,
                [&mut a, &mut b],
                Bf16GemmLayout::contiguous(5, 1024, 2048)
            )
            .is_err()
    );
    let mut alias = slice(&a, 0, 512);
    assert!(
        owner
            .execute_bf16_pair_view(&op, p, &resident, &x, [&mut a, &mut alias], layout)
            .is_err()
    );
    let foreign = CudaOperators::new_on_device(0).unwrap();
    assert!(
        owner
            .decode_numeric_bf16_into(&foreign, &w, &mut dst)
            .is_err()
    );
    assert!(
        owner
            .execute_bf16_pair_view(&foreign, p, &resident, &x, [&mut a, &mut b], layout)
            .is_err()
    );
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&foreign, [&w, &w], &mut resident)
            .is_err()
    );
    let foreign_w = foreign
        .upload_numeric_fp8_linear(
            l,
            &vec![0x38; l.n * l.k],
            &[0x80, 0x3f].repeat(l.n * l.k / 16384),
        )
        .unwrap();
    let mut foreign_target = foreign.zero_bf16_buffer(1024 * 2048).unwrap();
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&w, &foreign_w], &mut resident)
            .is_err()
    );
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&foreign_w, &w], &mut resident)
            .is_err()
    );
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&w, &w], &mut foreign_target)
            .is_err()
    );
    let foreign_x = foreign.zero_bf16_buffer(2048).unwrap();
    assert!(
        owner
            .execute_bf16_pair_view(&op, p, &resident, &foreign_x, [&mut a, &mut b], layout)
            .is_err()
    );
    let stream = op.stream_clone();
    stream.begin_capture().unwrap();
    assert!(owner.decode_numeric_bf16_into(&op, &w, &mut dst).is_err());
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&w, &w], &mut resident)
            .is_err()
    );
    assert!(
        owner
            .execute_bf16_pair_view(&op, p, &resident, &x, [&mut a, &mut b], layout)
            .is_err()
    );
    drop(stream.end_capture().unwrap());
    assert!(!owner.is_poisoned());
    owner.native.poisoned.set(true);
    assert!(owner.decode_numeric_bf16_into(&op, &w, &mut dst).is_err());
    assert!(
        owner
            .decode_numeric_bf16_pair_into(&op, [&w, &w], &mut resident)
            .is_err()
    );
    assert!(
        owner
            .execute_bf16_pair_view(&op, p, &resident, &x, [&mut a, &mut b], layout)
            .is_err()
    );
    // No submission occurred in this fault simulation; do not leak the fixture.
    owner.native.poisoned.set(false);
}
