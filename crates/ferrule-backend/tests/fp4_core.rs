#![cfg(feature = "cuda")]

//! Architecture-neutral FP4 core operator tests.

use ferrule_backend::cuda::context::CudaArtifactOperatorContext;
use ferrule_backend::cuda::kernels::kernels;
use ferrule_backend::cuda::runtime::{CudaContext, DeviceBuffer, LaunchConfig};
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("CUDA core operator test lock poisoned")
}

fn fp4_e2m1(nibble: u8) -> f32 {
    let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
    let magnitude = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    sign * magnitude
}

fn e8m0_scale(byte: u8) -> f32 {
    2.0f32.powi(byte as i32 - 127)
}

fn e8m0_scale_byte_for_amax(amax: f32, quant_max: f32) -> u8 {
    if !amax.is_finite() || amax <= 0.0 || !quant_max.is_finite() || quant_max <= 0.0 {
        return 127;
    }
    ((amax / quant_max).log2().ceil() as i32 + 127).clamp(0, 255) as u8
}

fn quantize_fp4_e2m1_nibble(value: f32) -> u8 {
    if !value.is_finite() || value == 0.0 {
        return 0;
    }
    let sign = if value < 0.0 { 0x08 } else { 0 };
    let magnitude = value.abs().min(6.0);
    let mut best = 0u8;
    let mut best_error = magnitude;
    for candidate in 1..8u8 {
        let error = (fp4_e2m1(candidate) - magnitude).abs();
        if error < best_error {
            best_error = error;
            best = candidate;
        }
    }
    sign | best
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

fn normalized_hadamard(values: &mut [f32]) {
    let mut span = 1;
    while span < values.len() {
        for start in (0..values.len()).step_by(span * 2) {
            for offset in 0..span {
                let left = start + offset;
                let right = left + span;
                let first = values[left];
                let second = values[right];
                values[left] = first + second;
                values[right] = first - second;
            }
        }
        span *= 2;
    }
    let scale = (values.len() as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
}

fn fp4_qat_in_place(values: &mut [f32], block_size: usize) {
    for block in values.chunks_exact_mut(block_size) {
        let amax = block
            .iter()
            .map(|value| value.abs())
            .fold(6.0 * 2.0f32.powi(-126), f32::max);
        let scale = e8m0_scale(e8m0_scale_byte_for_amax(amax, 6.0));
        for value in block {
            *value = fp4_e2m1(quantize_fp4_e2m1_nibble(*value / scale)) * scale;
        }
    }
}

#[test]
fn hadamard_fp4_qat_restores_bf16_before_scale_selection() {
    let _guard = cuda_test_guard();
    const COLUMNS: usize = 64;
    const BLOCK: usize = 32;

    let mut input = (0..COLUMNS)
        .map(|index| {
            if index.is_multiple_of(2) {
                0.9375
            } else {
                0.5625
            }
        })
        .collect::<Vec<_>>();
    input[0] = 0.941_406_25;

    let mut transformed = input.clone();
    normalized_hadamard(&mut transformed);
    assert_eq!(
        e8m0_scale_byte_for_amax(
            transformed[..BLOCK]
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f32::max),
            6.0
        ),
        128,
        "the fixture must cross the E8M0 scale boundary before BF16 writeback"
    );
    transformed
        .iter_mut()
        .for_each(|value| *value = bf16_round(*value));
    assert_eq!(
        e8m0_scale_byte_for_amax(
            transformed[..BLOCK]
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f32::max),
            6.0
        ),
        127
    );
    fp4_qat_in_place(&mut transformed, BLOCK);

    let context = CudaArtifactOperatorContext::new().expect("create CUDA operator context");
    let mut actual = context
        .upload_f32_buffer(&input)
        .expect("upload indexer values");
    context
        .fp4_hadamard_qat_quantize_buffer_in_place(&mut actual, COLUMNS)
        .expect("run Hadamard/FP4 QAT");
    let actual = context
        .download_f32_buffer(&actual)
        .expect("download QAT values");
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        transformed
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
#[allow(
    unsafe_code,
    reason = "the native operator facade requires an explicit acknowledgement of launch safety"
)]
fn fp4_activation_pack_matches_cpu_reference() {
    let _guard = cuda_test_guard();
    let Ok(context) = CudaContext::new(0) else {
        eprintln!("skipping FP4 core test: no CUDA device");
        return;
    };
    context.bind_to_thread().expect("bind CUDA context");
    let module = kernels::load(&context).expect("load native core provider");
    let stream = context.default_stream();

    const ROWS: usize = 2;
    const COLUMNS: usize = 64;
    const BLOCK: usize = 32;
    let values = (0..ROWS * COLUMNS)
        .map(|index| {
            let centered = (index as i32 % 17) - 8;
            centered as f32 * 0.375 + if index.is_multiple_of(11) { 4.25 } else { 0.0 }
        })
        .collect::<Vec<_>>();

    let mut expected_packed = vec![0u8; ROWS * (COLUMNS / 2)];
    let mut expected_scales = vec![0u8; ROWS * (COLUMNS / BLOCK)];
    for row in 0..ROWS {
        for block in 0..COLUMNS / BLOCK {
            let start = row * COLUMNS + block * BLOCK;
            let end = start + BLOCK;
            let amax = values[start..end]
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max);
            let scale_byte = e8m0_scale_byte_for_amax(amax, 6.0);
            let scale = e8m0_scale(scale_byte);
            expected_scales[row * (COLUMNS / BLOCK) + block] = scale_byte;
            for column in (0..BLOCK).step_by(2) {
                let low = quantize_fp4_e2m1_nibble(values[start + column] / scale);
                let high = quantize_fp4_e2m1_nibble(values[start + column + 1] / scale);
                expected_packed[row * (COLUMNS / 2) + block * (BLOCK / 2) + column / 2] =
                    low | (high << 4);
            }
        }
    }

    let values_device = DeviceBuffer::from_host(&stream, &values).expect("upload values");
    let mut packed_device =
        DeviceBuffer::<u8>::zeroed(&stream, expected_packed.len()).expect("allocate packed output");
    let mut scales_device =
        DeviceBuffer::<u8>::zeroed(&stream, expected_scales.len()).expect("allocate scale output");
    unsafe {
        module.fp4_e2m1_e8m0_quantize_f32_packed(
            &stream,
            LaunchConfig::for_num_elems((ROWS * (COLUMNS / BLOCK)) as u32),
            &values_device,
            &mut packed_device,
            &mut scales_device,
            0,
            values.len() as u32,
            COLUMNS as u32,
            BLOCK as u32,
        )
    }
    .expect("launch FP4 pack");

    assert_eq!(
        scales_device.to_host_vec(&stream).expect("download scales"),
        expected_scales
    );
    assert_eq!(
        packed_device.to_host_vec(&stream).expect("download packed"),
        expected_packed
    );
}
