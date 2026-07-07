//! FP4 microscaled (mxf4) warp-MMA smoke test for sm_120a/sm_121a (GB10).
//!
//! Verifies `fp4_mxf4_smoke` (mma.sync kind::mxf4.block_scale) against a
//! scalar FP4 (E2M1) × E8M0-block-scale reference for one 16×8×64 tile.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_cuda::kernels::kernels;
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("CUDA smoke test lock poisoned")
}

fn fp4_e2m1(nibble: u8) -> f32 {
    let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
    let mag = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    sign * mag
}

fn e8m0_scale(byte: u8) -> f32 {
    libm::ldexpf(1.0, byte as i32 - 127)
}

fn e8m0_scale_byte_for_amax(amax: f32, quant_max: f32) -> u8 {
    if !amax.is_finite() || amax <= 0.0 || !quant_max.is_finite() || quant_max <= 0.0 {
        return 127;
    }
    let byte = libm::ceilf(libm::log2f(amax / quant_max)) as i32 + 127;
    byte.clamp(0, 255) as u8
}

fn quantize_fp4_e2m1_nibble(value: f32) -> u8 {
    if !value.is_finite() || value == 0.0 {
        return 0;
    }
    let sign = if value < 0.0 { 0x08 } else { 0 };
    let magnitude = if value < 0.0 { -value } else { value }.min(6.0);
    let mut best = 0u8;
    let mut best_err = magnitude;
    for idx in 1..8u8 {
        let candidate = fp4_e2m1(idx);
        let err = (candidate - magnitude).abs();
        if err < best_err {
            best_err = err;
            best = idx;
        }
    }
    sign | best
}

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

fn pack_low_first(elems: &[u8]) -> Vec<u8> {
    assert!(elems.len().is_multiple_of(2));
    let mut packed = vec![0u8; elems.len() / 2];
    for (i, &nibble) in elems.iter().enumerate() {
        if i % 2 == 0 {
            packed[i / 2] = nibble & 0x0f;
        } else {
            packed[i / 2] |= (nibble & 0x0f) << 4;
        }
    }
    packed
}

fn all_ones_packed_tile() -> (Vec<u8>, Vec<u8>) {
    // FP4 E2M1 nibble 0b0010 is +1.0. Pack two values per byte.
    let a_packed = vec![0x22u8; 16 * (64 / 2)];
    let b_packed = vec![0x22u8; 8 * (64 / 2)];
    (a_packed, b_packed)
}

fn infer_probe_lane(value: f32, base_byte: u8) -> Option<usize> {
    let mut best_lane = None;
    let mut best_err = f32::INFINITY;
    for lane in 0..32usize {
        let scale = e8m0_scale(base_byte + lane as u8);
        let expected = 64.0 * scale;
        let err = (value - expected).abs();
        if err < best_err {
            best_err = err;
            best_lane = Some(lane);
        }
    }
    let lane = best_lane?;
    let expected = 64.0 * e8m0_scale(base_byte + lane as u8);
    let tol = (expected.abs() * 1e-4).max(1e-3);
    if (value - expected).abs() <= tol {
        Some(lane)
    } else {
        None
    }
}

fn format_lane_matrix(values: &[Option<usize>]) -> String {
    let mut out = String::new();
    for r in 0..16 {
        for c in 0..8 {
            match values[r * 8 + c] {
                Some(lane) => out.push_str(&format!("{lane:2} ")),
                None => out.push_str("?? "),
            }
        }
        out.push('\n');
    }
    out
}

#[test]
fn fp4_mxf4_matches_scalar_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let ctx = CudaContext::new(0).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let module = kernels::load(&ctx).unwrap_or_else(|e| panic!("module load failed: {e:?}"));
    let stream = ctx.default_stream();

    const M: usize = 16;
    const N: usize = 8;
    const K: usize = 64;
    const BLOCK: usize = 32;
    const NBLOCKS: usize = K / BLOCK;

    // A: M×K row-major FP4 packed (K/2 = 32 bytes/row).
    let mut a_packed = vec![0u8; M * (K / 2)];
    let mut a_elems = vec![0u8; M * K];
    for r in 0..M {
        for c in 0..K {
            let mag = ((r + c) % 6) as u8;
            let sign = if (r * K + c) % 3 == 0 { 0x08 } else { 0 };
            let nibble = mag | sign;
            a_elems[r * K + c] = nibble;
            if c % 2 == 0 {
                a_packed[r * (K / 2) + c / 2] = nibble;
            } else {
                a_packed[r * (K / 2) + c / 2] |= nibble << 4;
            }
        }
    }
    let mut a_scales = vec![0u8; M * NBLOCKS];
    for r in 0..M {
        a_scales[r * NBLOCKS] = 127;
        a_scales[r * NBLOCKS + 1] = 128;
    }

    // B: K×N col-major FP4 packed (K/2 = 32 bytes/col).
    let mut b_packed = vec![0u8; N * (K / 2)];
    let mut b_elems = vec![0u8; K * N];
    for col in 0..N {
        for r in 0..K {
            let mag = ((r + col + 1) % 6) as u8;
            let sign = if (r * N + col) % 4 == 0 { 0x08 } else { 0 };
            let nibble = mag | sign;
            b_elems[col * K + r] = nibble;
            if r % 2 == 0 {
                b_packed[col * (K / 2) + r / 2] = nibble;
            } else {
                b_packed[col * (K / 2) + r / 2] |= nibble << 4;
            }
        }
    }
    let mut b_scales = vec![0u8; N * NBLOCKS];
    for col in 0..N {
        b_scales[col * NBLOCKS] = 127;
        b_scales[col * NBLOCKS + 1] = 126;
    }

    // Scalar reference: the kernel uses scale_a/scale_b from row 0 / col 0 only.
    let sa0 = e8m0_scale(a_scales[0]);
    let sa1 = e8m0_scale(a_scales[1]);
    let sb0 = e8m0_scale(b_scales[0]);
    let sb1 = e8m0_scale(b_scales[1]);
    let mut kernel_ref = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for blk in 0..NBLOCKS {
                let (sa, sb) = if blk == 0 { (sa0, sb0) } else { (sa1, sb1) };
                let mut block_dot = 0.0f32;
                for j in 0..BLOCK {
                    let k = blk * BLOCK + j;
                    block_dot += fp4_e2m1(a_elems[m * K + k]) * fp4_e2m1(b_elems[n * K + k]);
                }
                acc += block_dot * sa * sb;
            }
            kernel_ref[m * N + n] = acc;
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a_packed).expect("A");
    let b_dev = DeviceBuffer::from_host(&stream, &b_packed).expect("B");
    let as_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("As");
    let bs_dev = DeviceBuffer::from_host(&stream, &b_scales).expect("Bs");
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, M * N).expect("C");

    unsafe {
        module.fp4_mxf4_smoke(
            &stream,
            LaunchConfig::for_num_elems(32),
            &a_dev,
            &b_dev,
            &as_dev,
            &bs_dev,
            &mut c_dev,
        )
    }
    .expect("launch");

    let got = c_dev.to_host_vec(&stream).expect("download");
    stream.synchronize().expect("sync");

    let mut max_err = 0.0f32;
    for i in 0..M * N {
        max_err = max_err.max((got[i] - kernel_ref[i]).abs());
    }
    let tol = 1e-2;
    assert!(
        max_err <= tol,
        "FP4 mxf4 mismatch: max_err={max_err} (tol={tol})\ngot={got:?}\nref={kernel_ref:?}"
    );
    println!("FP4 mxf4 smoke OK: max_err={max_err} (tol={tol})");
}

#[test]
fn fp4_mxf4_full_tile_matches_per_row_col_scale_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let ctx = CudaContext::new(0).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let module = kernels::load(&ctx).unwrap_or_else(|e| panic!("module load failed: {e:?}"));
    let stream = ctx.default_stream();

    const M: usize = 16;
    const N: usize = 8;
    const K: usize = 64;
    const BLOCK: usize = 32;
    const NBLOCKS: usize = K / BLOCK;

    let mut a_elems = vec![0u8; M * K];
    for r in 0..M {
        for c in 0..K {
            let mag = ((r * 5 + c * 3 + 1) % 6) as u8;
            let sign = if (r * 11 + c) % 9 == 0 { 0x08 } else { 0 };
            a_elems[r * K + c] = mag | sign;
        }
    }
    let a_packed = pack_low_first(&a_elems);

    let mut b_packed = vec![0u8; N * (K / 2)];
    let mut b_elems = vec![0u8; N * K];
    for col in 0..N {
        let mut col_elems = vec![0u8; K];
        for k in 0..K {
            let mag = ((col * 7 + k * 5 + 2) % 6) as u8;
            let sign = if (col + k * 3) % 10 == 0 { 0x08 } else { 0 };
            let nibble = mag | sign;
            col_elems[k] = nibble;
            b_elems[col * K + k] = nibble;
        }
        let packed_col = pack_low_first(&col_elems);
        b_packed[col * (K / 2)..(col + 1) * (K / 2)].copy_from_slice(&packed_col);
    }

    let mut a_scales = vec![0u8; M * NBLOCKS];
    for r in 0..M {
        a_scales[r * NBLOCKS] = 126 + (r % 3) as u8;
        a_scales[r * NBLOCKS + 1] = 127 + ((r + 1) % 3) as u8;
    }
    let mut b_scales = vec![0u8; N * NBLOCKS];
    for col in 0..N {
        b_scales[col * NBLOCKS] = 127 + (col % 2) as u8;
        b_scales[col * NBLOCKS + 1] = 126 + ((col + 1) % 3) as u8;
    }

    let mut expected = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for blk in 0..NBLOCKS {
                let scale = e8m0_scale(a_scales[m * NBLOCKS + blk])
                    * e8m0_scale(b_scales[n * NBLOCKS + blk]);
                let mut block_dot = 0.0f32;
                for j in 0..BLOCK {
                    let k = blk * BLOCK + j;
                    block_dot += fp4_e2m1(a_elems[m * K + k]) * fp4_e2m1(b_elems[n * K + k]);
                }
                acc += block_dot * scale;
            }
            expected[m * N + n] = acc;
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a_packed).expect("A");
    let b_dev = DeviceBuffer::from_host(&stream, &b_packed).expect("B");
    let as_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("As");
    let bs_dev = DeviceBuffer::from_host(&stream, &b_scales).expect("Bs");
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, M * N).expect("C");

    unsafe {
        module.fp4_mxf4_full_tile(
            &stream,
            LaunchConfig::for_num_elems(32),
            &a_dev,
            &b_dev,
            &as_dev,
            &bs_dev,
            &mut c_dev,
        )
    }
    .expect("launch");

    let got = c_dev.to_host_vec(&stream).expect("download");
    stream.synchronize().expect("sync");
    let mut max_err = 0.0f32;
    for i in 0..M * N {
        max_err = max_err.max((got[i] - expected[i]).abs());
    }
    let tol = 1e-2;
    assert!(
        max_err <= tol,
        "FP4 mxf4 full tile mismatch: max_err={max_err} (tol={tol})\ngot={got:?}\nref={expected:?}"
    );
    println!("FP4 mxf4 full tile OK: max_err={max_err} (tol={tol})");
}

#[test]
fn fp4_activation_pack_kernel_matches_cpu_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let ctx = CudaContext::new(0).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let module = kernels::load(&ctx).unwrap_or_else(|e| panic!("module load failed: {e:?}"));
    let stream = ctx.default_stream();

    const ROWS: usize = 2;
    const COLS: usize = 64;
    const BLOCK: usize = 32;
    let values = (0..ROWS * COLS)
        .map(|i| {
            let centered = (i as i32 % 17) - 8;
            centered as f32 * 0.375 + if i % 11 == 0 { 4.25 } else { 0.0 }
        })
        .collect::<Vec<_>>();

    let mut expected_packed = vec![0u8; ROWS * (COLS / 2)];
    let mut expected_scales = vec![0u8; ROWS * (COLS / BLOCK)];
    for row in 0..ROWS {
        for block in 0..COLS / BLOCK {
            let start = row * COLS + block * BLOCK;
            let end = start + BLOCK;
            let amax = values[start..end]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            let scale_byte = e8m0_scale_byte_for_amax(amax, 6.0);
            let scale = e8m0_scale(scale_byte);
            expected_scales[row * (COLS / BLOCK) + block] = scale_byte;
            for j in (0..BLOCK).step_by(2) {
                let n0 = quantize_fp4_e2m1_nibble(values[start + j] / scale);
                let n1 = quantize_fp4_e2m1_nibble(values[start + j + 1] / scale);
                expected_packed[row * (COLS / 2) + block * (BLOCK / 2) + j / 2] = n0 | (n1 << 4);
            }
        }
    }

    let values_dev = DeviceBuffer::from_host(&stream, &values).expect("values");
    let mut packed_dev =
        DeviceBuffer::<u8>::zeroed(&stream, expected_packed.len()).expect("packed");
    let mut scales_dev =
        DeviceBuffer::<u8>::zeroed(&stream, expected_scales.len()).expect("scales");
    unsafe {
        module.fp4_e2m1_e8m0_quantize_f32_packed(
            &stream,
            LaunchConfig::for_num_elems((ROWS * (COLS / BLOCK)) as u32),
            &values_dev,
            &mut packed_dev,
            &mut scales_dev,
            values.len() as u32,
            COLS as u32,
            BLOCK as u32,
        )
    }
    .expect("launch");

    let got_packed = packed_dev.to_host_vec(&stream).expect("packed download");
    let got_scales = scales_dev.to_host_vec(&stream).expect("scales download");
    stream.synchronize().expect("sync");
    assert_eq!(got_scales, expected_scales, "FP4 activation scale bytes");
    assert_eq!(got_packed, expected_packed, "FP4 activation packed bytes");
    println!("FP4 activation pack OK: {} values", values.len());
}

#[test]
fn fp4_mxf4_gemv8_tile_matches_scalar_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let ctx = CudaContext::new(0).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let module = kernels::load(&ctx).unwrap_or_else(|e| panic!("module load failed: {e:?}"));
    let stream = ctx.default_stream();

    const M: usize = 8;
    const K: usize = 64;
    const BLOCK: usize = 32;
    const NBLOCKS: usize = K / BLOCK;

    let mut a_elems = vec![0u8; M * K];
    for r in 0..M {
        for c in 0..K {
            let mag = ((r * 3 + c * 5 + 2) % 6) as u8;
            let sign = if (r + c) % 7 == 0 { 0x08 } else { 0 };
            a_elems[r * K + c] = mag | sign;
        }
    }
    let a_packed = pack_low_first(&a_elems);

    let mut x_elems = vec![0u8; K];
    for c in 0..K {
        let mag = ((c * 3 + 1) % 6) as u8;
        let sign = if c % 5 == 0 { 0x08 } else { 0 };
        x_elems[c] = mag | sign;
    }
    let x_packed = pack_low_first(&x_elems);

    let mut a_scales = vec![0u8; M * NBLOCKS];
    for r in 0..M {
        a_scales[r * NBLOCKS] = 126 + (r % 3) as u8;
        a_scales[r * NBLOCKS + 1] = 127 + ((r + 1) % 3) as u8;
    }
    let x_scales = vec![128u8, 126u8];

    let mut expected = vec![0.0f32; M];
    for r in 0..M {
        let mut acc = 0.0f32;
        for blk in 0..NBLOCKS {
            let scale = e8m0_scale(a_scales[r * NBLOCKS + blk]) * e8m0_scale(x_scales[blk]);
            let mut block_dot = 0.0f32;
            for j in 0..BLOCK {
                let k = blk * BLOCK + j;
                block_dot += fp4_e2m1(a_elems[r * K + k]) * fp4_e2m1(x_elems[k]);
            }
            acc += block_dot * scale;
        }
        expected[r] = acc;
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a_packed).expect("A");
    let x_dev = DeviceBuffer::from_host(&stream, &x_packed).expect("X");
    let as_dev = DeviceBuffer::from_host(&stream, &a_scales).expect("As");
    let xs_dev = DeviceBuffer::from_host(&stream, &x_scales).expect("Xs");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, M).expect("out");

    unsafe {
        module.fp4_mxf4_gemv8_tile(
            &stream,
            LaunchConfig::for_num_elems(32),
            &a_dev,
            &x_dev,
            &as_dev,
            &xs_dev,
            &mut out_dev,
        )
    }
    .expect("launch");

    let got = out_dev.to_host_vec(&stream).expect("download");
    stream.synchronize().expect("sync");
    let mut max_err = 0.0f32;
    for i in 0..M {
        max_err = max_err.max((got[i] - expected[i]).abs());
    }
    let tol = 1e-2;
    assert!(
        max_err <= tol,
        "FP4 mxf4 gemv8 mismatch: max_err={max_err} (tol={tol})\ngot={got:?}\nref={expected:?}"
    );
    println!("FP4 mxf4 gemv8 tile OK: max_err={max_err} (tol={tol})");
}

#[test]
fn fp4_mxf4_scale_lane_selector_probe() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let ctx = CudaContext::new(0).expect("ctx");
    ctx.bind_to_thread().expect("bind");
    let module = kernels::load(&ctx).unwrap_or_else(|e| panic!("module load failed: {e:?}"));
    let stream = ctx.default_stream();

    const M: usize = 16;
    const N: usize = 8;
    const BASE: u8 = 120;
    let (a_packed, b_packed) = all_ones_packed_tile();
    let a_dev = DeviceBuffer::from_host(&stream, &a_packed).expect("A");
    let b_dev = DeviceBuffer::from_host(&stream, &b_packed).expect("B");

    for vary_a in [true, false] {
        let mut lane_a_scales = vec![127u8; 32 * 2];
        let mut lane_b_scales = vec![127u8; 32 * 2];
        for lane in 0..32usize {
            let byte = BASE + lane as u8;
            let scales = if vary_a {
                &mut lane_a_scales
            } else {
                &mut lane_b_scales
            };
            scales[lane * 2] = byte;
            scales[lane * 2 + 1] = byte;
        }
        let as_dev = DeviceBuffer::from_host(&stream, &lane_a_scales).expect("As");
        let bs_dev = DeviceBuffer::from_host(&stream, &lane_b_scales).expect("Bs");
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, M * N).expect("C");

        unsafe {
            module.fp4_mxf4_scale_lane_probe(
                &stream,
                LaunchConfig::for_num_elems(32),
                &a_dev,
                &b_dev,
                &as_dev,
                &bs_dev,
                &mut c_dev,
            )
        }
        .expect("launch");

        let got = c_dev.to_host_vec(&stream).expect("download");
        stream.synchronize().expect("sync");
        let lanes = got
            .iter()
            .map(|&value| infer_probe_lane(value, BASE))
            .collect::<Vec<_>>();
        let inferred = lanes.iter().filter(|lane| lane.is_some()).count();
        assert!(
            inferred > 0,
            "could not infer any lanes for {} scale probe\ngot={got:?}\nlanes:\n{}",
            if vary_a { "A" } else { "B" },
            format_lane_matrix(&lanes)
        );
        println!(
            "mxf4 {} scale selector lane map ({inferred}/{} inferred):\n{}",
            if vary_a { "A" } else { "B" },
            lanes.len(),
            format_lane_matrix(&lanes)
        );
    }
}
