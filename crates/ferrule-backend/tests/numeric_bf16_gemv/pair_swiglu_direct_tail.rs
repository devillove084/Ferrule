//! Native formula word regression, not HF equivalence (especially nonfinite).
//! Fixed words supplement an independently frozen device expression, not the
//! production arithmetic helper. Inputs bypass GEMM and all float upload casts.
const WIDTH: usize = 256;
const AXIS: [u16; 15] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x4301, 0x7f80, 0xff80, 0x7fc1,
    0xffc1, 0x7f81, 0xff81,
];
// (gate, up, activation). Exact zero/subnormal halves, saturated SiLU, product
// midpoints (129/131 * +/-1.5), and non-invalid infinity cases; no CPU exp oracle.
const FIXED: [(u16, u16, u16); 24] = [
    (0x0000, 0x3f80, 0x0000),
    (0x0000, 0xbf80, 0x8000),
    (0x8000, 0x3f80, 0x8000),
    (0x8000, 0xbf80, 0x0000),
    (0x0001, 0x3f80, 0x0000),
    (0x8001, 0x3f80, 0x8000),
    (0x0003, 0x3f80, 0x0002),
    (0x8003, 0x3f80, 0x8002),
    (0x007f, 0x3f80, 0x0040),
    (0x807f, 0x3f80, 0x8040),
    (0x0080, 0x3f80, 0x0040),
    (0x8080, 0x3f80, 0x8040),
    (0x4300, 0x0001, 0x0080),
    (0x4300, 0x8001, 0x8080),
    (0x4301, 0x3fc0, 0x4342),
    (0x4303, 0x3fc0, 0x4344),
    (0x4301, 0xbfc0, 0xc342),
    (0x4303, 0xbfc0, 0xc344),
    (0xc300, 0x3f80, 0x8000),
    (0xc300, 0xbf80, 0x0000),
    (0x7f80, 0x3f80, 0x7f80),
    (0x7f80, 0xbf80, 0xff80),
    (0xbf80, 0x7f80, 0xff80),
    (0xbf80, 0xff80, 0x7f80),
];
fn case(index: usize) -> (u16, u16, Option<u16>) {
    let index = index % (FIXED.len() + AXIS.len() * AXIS.len());
    if index < FIXED.len() {
        let (g, u, y) = FIXED[index];
        (g, u, Some(y))
    } else {
        let i = index - FIXED.len();
        (AXIS[i / AXIS.len()], AXIS[i % AXIS.len()], None)
    }
}
fn cases(rows: usize) -> Vec<(u16, u16, Option<u16>)> {
    (0..rows)
        .flat_map(|r| (0..WIDTH).map(move |c| case(c + 37 * r)))
        .collect()
}
fn packed(cases: &[(u16, u16, Option<u16>)]) -> Vec<u16> {
    cases
        .as_chunks::<WIDTH>()
        .0
        .iter()
        .flat_map(|row| row.iter().map(|c| c.0).chain(row.iter().map(|c| c.1)))
        .collect()
}
fn rne(bits: u32) -> u16 {
    let mut word = (bits >> 16) as u16;
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return word | 0x0040;
    }
    let residual = bits & 0xffff;
    if residual > 0x8000 || (residual == 0x8000 && word & 1 != 0) {
        word = word.wrapping_add(1);
    }
    word
}

#[test]
fn direct_tail_fixture_has_signed_zero_nonfinite_and_pinned_ties() {
    assert!(FIXED.len() + AXIS.len() * AXIS.len() <= WIDTH);
    for rows in [1, 2, 5] {
        let cases = cases(rows);
        let packed = packed(&cases);
        for (i, &(g, u, _)) in cases.iter().enumerate() {
            let base = i / WIDTH * 2 * WIDTH + i % WIDTH;
            assert_eq!(packed[base], g);
            assert_eq!(packed[base + WIDTH], u);
        }
        for g in AXIS {
            for u in AXIS {
                assert!(cases.iter().any(|c| (c.0, c.1) == (g, u)));
            }
        }
        for word in [0x0000, 0x8000, 0x0002, 0x8002, 0x7f80, 0xff80] {
            assert!(cases.iter().any(|c| c.2 == Some(word)));
        }
    }
    // Pin both tie directions/signs and explicit NaN quiet-bit conversion.
    for (raw, expected) in [
        (0x0000_8000, 0x0000),
        (0x8000_8000, 0x8000),
        (0x0001_8000, 0x0002),
        (0x8001_8000, 0x8002),
        (0x4341_8000, 0x4342),
        (0x4344_8000, 0x4344),
        (0xc341_8000, 0xc342),
        (0xc344_8000, 0xc344),
        (0x7f81_0000, 0x7fc1),
        (0xff81_0000, 0xffc1),
    ] {
        assert_eq!(rne(raw), expected, "raw={raw:08x}");
    }
}

#[cfg(all(feature = "cuda", ferrule_cuda_test_oracle))]
#[allow(unsafe_code)]
mod gpu {
    use super::*;
    use ferrule_backend::cuda::{
        operators::linear::{CudaBf16Buffer, CudaOperators},
        providers::DeviceBuffer,
    };
    #[repr(C)]
    struct TailArgs {
        rows: u32,
        width: u32,
        packed: u64,
        gate: u64,
        up: u64,
        reference: u64,
        fused: u64,
        raw: u64,
        stream: u64,
    }
    unsafe extern "C" {
        fn ferrule_test_pair_swiglu_tails(args: *const TailArgs) -> i32;
    }
    fn words(op: &CudaOperators, b: &CudaBf16Buffer) -> Vec<u16> {
        b.as_device_buffer()
            .to_host_vec(&op.stream_clone())
            .unwrap()
    }
    #[test]
    #[ignore = "sole GPU owner; opt-in direct-tail integer-word native oracle, NOT HF nonfinite equivalence"]
    fn direct_tail_words_old_literal_vs_fused_m1_m2_m5() {
        assert_eq!(std::mem::size_of::<TailArgs>(), 64);
        let op = CudaOperators::new_on_device(0).unwrap();
        let stream = op.stream_clone();
        for rows in [1, 2, 5] {
            let cases = cases(rows);
            let input_words = packed(&cases);
            let input = op.upload_bf16_words(&input_words).unwrap();
            let count = rows * WIDTH;
            let gate = op.zero_bf16_buffer(count).unwrap();
            let up = op.zero_bf16_buffer(count).unwrap();
            let reference = op.zero_bf16_buffer(count).unwrap();
            let fused = op.zero_bf16_buffer(count).unwrap();
            let raw = DeviceBuffer::<u32>::zeroed(&stream, 2 * count).unwrap();
            let args = TailArgs {
                rows: rows as u32,
                width: WIDTH as u32,
                packed: input.as_device_buffer().cu_deviceptr(),
                gate: gate.as_device_buffer().cu_deviceptr(),
                up: up.as_device_buffer().cu_deviceptr(),
                reference: reference.as_device_buffer().cu_deviceptr(),
                fused: fused.as_device_buffer().cu_deviceptr(),
                raw: raw.cu_deviceptr(),
                stream: stream.cu_stream() as usize as u64,
            };
            stream.context().bind_to_thread().unwrap();
            // SAFETY: distinct exact-sized buffers on this bound context/stream;
            // all allocations remain alive through the downloads below.
            assert_eq!(unsafe { ferrule_test_pair_swiglu_tails(&args) }, 0);
            // The current old API must agree with the independent frozen literal
            // too; the literal, not this shared-helper path, is the oracle.
            let boundary = op.swiglu_bf16(&gate, &up).unwrap();
            let g = words(&op, &gate);
            let u = words(&op, &up);
            let expected = words(&op, &reference);
            let actual = words(&op, &fused);
            let boundary = words(&op, &boundary);
            let raw = raw.to_host_vec(&stream).unwrap();
            assert_eq!(words(&op, &input), input_words);
            for (i, &(gate, up, fixed)) in cases.iter().enumerate() {
                assert_eq!((g[i], u[i]), (gate, up), "split M{rows} word{i}");
                assert_eq!(
                    expected[i],
                    rne(raw[2 * i + 1]),
                    "raw product M{rows} word{i}"
                );
                for (label, value) in [("new tail", actual[i]), ("old boundary", boundary[i])] {
                    assert_eq!(
                        value, expected[i],
                        "{label} M{rows} word{i} gate={gate:04x} up={up:04x}, actual={value:04x} expected={:04x}",
                        expected[i]
                    );
                    if let Some(fixed) = fixed {
                        assert_eq!(value, fixed, "fixed {label} M{rows} word{i}");
                    }
                }
                if gate & 0x7fff > 0x7f80
                    || up & 0x7fff > 0x7f80
                    || gate == 0xff80
                    || (gate & 0x7fff == 0 && up & 0x7fff == 0x7f80)
                    || (gate == 0x7f80 && up & 0x7fff == 0)
                {
                    assert_eq!(expected[i] & 0x7fc0, 0x7fc0, "quiet NaN word{i}");
                }
            }
            for bits in [0x0000, 0x8000] {
                assert!(
                    actual.contains(&bits),
                    "M{rows} must produce explicit zero word {bits:04x}"
                );
            }
            // Prove these are actual midpoint stages, not just fixture labels.
            for (g, u, stage, bits) in [
                (0x0001, 0x3f80, 0, 0x0000_8000),
                (0x8001, 0x3f80, 0, 0x8000_8000),
                (0x0003, 0x3f80, 0, 0x0001_8000),
                (0x8003, 0x3f80, 0, 0x8001_8000),
                (0x4301, 0x3fc0, 1, 0x4341_8000),
                (0x4303, 0x3fc0, 1, 0x4344_8000),
                (0x4301, 0xbfc0, 1, 0xc341_8000),
                (0x4303, 0xbfc0, 1, 0xc344_8000),
            ] {
                let i = cases.iter().position(|c| (c.0, c.1) == (g, u)).unwrap();
                assert_eq!(raw[2 * i + stage], bits, "M{rows} raw tie {g:04x}/{u:04x}");
            }
        }
    }
}
