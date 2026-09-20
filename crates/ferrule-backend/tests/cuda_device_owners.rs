#![cfg(feature = "cuda")]

use std::error::Error;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread;

use ferrule_backend::cuda::operators::norm::CudaOperators;
use ferrule_backend::cuda::providers::CudaContext;

type TestResult = Result<(), Box<dyn Error>>;

fn require_devices(required: usize) {
    let visible = CudaContext::device_count().expect("enumerate visible CUDA devices");
    assert!(
        visible >= required,
        "requires at least {required} visible CUDA devices, found {visible}; check CUDA_VISIBLE_DEVICES"
    );
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

fn rms_case(ordinal: usize, width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input: Vec<_> = (0..width)
        .map(|index| {
            ((index * (ordinal + 3) + ordinal * 17) % 251) as f32 / 64.0 - 1.75
                + ordinal as f32 / 32.0
        })
        .collect();
    let weight: Vec<_> = (0..width)
        .map(|index| 0.5 + ((index + ordinal * 13) % 67) as f32 / 128.0)
        .collect();
    // Affine RMS uses F32 reduction and rounds only the final output to BF16.
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / width as f32;
    let inverse_rms = (mean_square + 1e-5).sqrt().recip();
    let expected = input
        .iter()
        .zip(&weight)
        .map(|(&value, &weight)| bf16_round(value * inverse_rms * weight))
        .collect();
    (input, weight, expected)
}

fn check_rms(ordinal: usize, actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "ordinal={ordinal} RMS length");
    let mut max_error = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite() && expected.is_finite(),
            "ordinal={ordinal} RMS[{index}] non-finite actual={actual} expected={expected}"
        );
        assert_eq!(
            actual.to_bits() & 0xffff,
            0,
            "RMS output must be BF16-rounded"
        );
        assert_eq!(
            expected.to_bits() & 0xffff,
            0,
            "reference must be BF16-rounded"
        );
        // Smoke allowance only, not an exact numerical contract: reduction/rsqrt
        // may cross one BF16 midpoint, but never accept two representable steps.
        let ulps = bf16_ordered_code(actual).abs_diff(bf16_ordered_code(expected));
        assert!(
            ulps <= 1,
            "ordinal={ordinal} RMS[{index}] actual={actual} expected={expected} error={error} BF16_ULPs={ulps}"
        );
        max_error = max_error.max(error);
    }
    max_error
}

// Monotonic BF16 encoding with +0 and -0 mapped to the same position.
fn bf16_ordered_code(value: f32) -> i32 {
    let word = (value.to_bits() >> 16) as u16;
    let magnitude = i32::from(word & 0x7fff);
    if word & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn synchronize_owner(operators: &CudaOperators) -> Result<(), String> {
    // Attempt both even when compute synchronization fails.
    let compute = operators.sync_stream().map_err(|error| error.to_string());
    let upload = operators
        .sync_upload_stream()
        .map_err(|error| error.to_string());
    if compute.is_ok() && upload.is_ok() {
        Ok(())
    } else {
        Err(format!(
            "ordinal={} compute sync={compute:?}; upload sync={upload:?}; cleanup completion not confirmed",
            operators.device_ordinal(),
        ))
    }
}

fn run_local_owner(ordinal: usize) -> TestResult {
    let operators = CudaOperators::new_on_device(ordinal)?;
    let work = catch_unwind(AssertUnwindSafe(
        || -> Result<(String, f32), Box<dyn Error>> {
            assert_eq!(operators.device_ordinal(), ordinal);
            let name = operators.stream_clone().context().device_name()?;
            let mut max_error = 0.0f32;
            for width in [257, 1024, 4096] {
                let (input, weight, expected) = rms_case(ordinal, width);
                let actual = operators.rms_norm(&input, &weight, 1e-5)?;
                max_error = max_error.max(check_rms(ordinal, &actual, &expected));
            }
            Ok((name, max_error))
        },
    ));
    let work = match work {
        Ok(result) => result.map_err(|error| error.to_string()),
        Err(payload) => Err(format!(
            "owner panicked: {}",
            panic_message(payload.as_ref())
        )),
    };
    let synchronized = synchronize_owner(&operators);
    drop(operators);
    match (work, synchronized) {
        (Ok((name, max_error)), Ok(())) => {
            println!(
                "owner ordinal={ordinal} name={name} max_abs_error={max_error:.8} synchronized/dropped"
            );
            Ok(())
        }
        (work, synchronized) => Err(format!(
            "ordinal={ordinal} work={work:?}; synchronization={synchronized:?}; owner failed"
        )
        .into()),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload")
}

#[test]
#[ignore = "requires at least eight real CUDA GPUs"]
fn eight_device_local_owners_smoke() {
    require_devices(8);
    let mut owners = Vec::new();
    let mut failures = Vec::new();
    for ordinal in 0..8 {
        // Only the ordinal crosses the thread boundary, never CUDA objects.
        match thread::Builder::new()
            .name(format!("cuda-owner-{ordinal}"))
            .spawn(move || run_local_owner(ordinal).map_err(|error| error.to_string()))
        {
            Ok(owner) => owners.push((ordinal, owner)),
            Err(error) => failures.push(format!("ordinal={ordinal} spawn failed: {error}")),
        }
    }
    // Do not fail fast: even initialization failures must join every spawned owner.
    for (ordinal, owner) in owners {
        match owner.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => failures.push(format!("ordinal={ordinal}: {error}")),
            Err(payload) => {
                let message = panic_message(payload.as_ref());
                failures.push(format!("ordinal={ordinal} panicked: {message}"));
            }
        }
    }
    assert!(failures.is_empty(), "CUDA owner failures: {failures:#?}");
}

#[test]
#[ignore = "requires at least two real CUDA GPUs"]
fn two_device_same_thread_context_switch_regression() -> TestResult {
    require_devices(2);
    let operators = [CudaOperators::new()?, CudaOperators::new_on_device(1)?];
    assert_eq!(operators[0].device_ordinal(), 0);
    assert_eq!(operators[1].device_ordinal(), 1);
    for ordinal in 0..2 {
        let owner = &operators[ordinal];
        let other_stream = operators[1 - ordinal].stream_clone();
        let switch_away = || other_stream.context().bind_to_thread();
        let name = owner.stream_clone().context().device_name()?;
        let values: Vec<_> = (0..257)
            .map(|index| index as f32 / 16.0 + ordinal as f32 + 1.0)
            .collect();
        let mut source = owner.upload_f32_buffer(&values)?;
        let mut destination = owner.zero_f32_buffer(values.len())?;
        owner.sync_stream()?;

        // Existing live allocations leave free allocator space, with no retired
        // events or driver allocation to accidentally bind the owner context.
        let before = owner.allocator_metrics();
        switch_away()?;
        let zeroed = owner.zero_f32_buffer(values.len())?;
        let after = owner.allocator_metrics();
        assert_eq!(after.driver_allocations, before.driver_allocations);
        assert!(after.reuse_allocations > before.reuse_allocations);
        switch_away()?;
        assert_eq!(owner.download_f32_buffer(&zeroed)?, vec![0.0; values.len()]);

        let replacement: Vec<_> = values.iter().map(|value| -value - 0.5).collect();
        assert_ne!(replacement, values);
        switch_away()?;
        owner.overwrite_f32_buffer(&replacement, &mut source)?;
        switch_away()?;
        assert_eq!(owner.download_f32_buffer(&source)?, replacement);

        switch_away()?;
        owner.copy_f32_range(&source, 0, &mut destination, 0, values.len())?;
        switch_away()?;
        assert_eq!(owner.download_f32_buffer(&destination)?, replacement);

        switch_away()?;
        owner.zero_f32_buffer_in_place(&mut destination)?;
        switch_away()?;
        assert_eq!(
            owner.download_f32_buffer(&destination)?,
            vec![0.0; values.len()]
        );

        switch_away()?;
        owner.overwrite_f32_range(&values[1..9], &mut destination, 3)?;
        switch_away()?;
        assert_eq!(owner.download_f32_range(&destination, 3, 8)?, values[1..9]);
        let mut expected_copy = vec![0.0; values.len()];
        expected_copy[3..11].copy_from_slice(&values[1..9]);
        switch_away()?;
        assert_eq!(owner.download_f32_buffer(&destination)?, expected_copy);

        let (input, weight, expected) = rms_case(ordinal, 1024);
        switch_away()?;
        let actual = owner.rms_norm(&input, &weight, 1e-5)?;
        let max_error = check_rms(ordinal, &actual, &expected);
        switch_away()?;
        synchronize_owner(owner)?;
        println!(
            "switch ordinal={ordinal} name={name} max_abs_error={max_error:.8} copies/memsets passed"
        );
    }
    drop(operators);
    Ok(())
}

#[test]
#[ignore = "requires at least two real CUDA GPUs"]
fn foreign_owner_buffers_are_rejected() -> TestResult {
    require_devices(2);
    let owner = CudaOperators::new_on_device(0)?;
    // A separately constructed owner on ordinal 0 is still foreign. Only Arc
    // clones/slices share an ordinary buffer owner; expert fences are separate APIs.
    for foreign_ordinal in [1, 0] {
        let foreign = CudaOperators::new_on_device(foreign_ordinal)?;
        let stream = owner.stream_clone();
        let foreign_stream = foreign.stream_clone();
        assert!(!Arc::ptr_eq(stream.context(), foreign_stream.context()));
        let values = vec![2.0f32; 16];
        let foreign_values = vec![3.0f32; 16];
        let sentinel = vec![-7.0f32; 16];
        let input = owner.upload_f32_buffer(&values)?;
        let weight = owner.upload_norm_weight(&values)?;
        let mut output = owner.upload_f32_buffer(&sentinel)?;
        let foreign_input = foreign.upload_f32_buffer(&foreign_values)?;
        let foreign_weight = foreign.upload_norm_weight(&foreign_values)?;
        let mut foreign_output = foreign.upload_f32_buffer(&sentinel)?;

        macro_rules! rejected {
            ($operation:expr) => {{
                let error = match $operation {
                    Ok(_) => panic!(
                        "foreign ordinal={foreign_ordinal}: {} accepted",
                        stringify!($operation)
                    ),
                    Err(error) => error,
                };
                assert!(error.to_string().contains("owner mismatch"), "{error}");
                // Check both owners after every rejection, not only at the end.
                assert_eq!(owner.download_f32_buffer(&output)?, sentinel);
                assert_eq!(foreign.download_f32_buffer(&foreign_output)?, sentinel);
                assert_eq!(owner.download_f32_buffer(&input)?, values);
                assert_eq!(foreign.download_f32_buffer(&foreign_input)?, foreign_values);
            }};
        }

        rejected!(owner.copy_f32_range(&foreign_input, 0, &mut output, 0, 16));
        rejected!(owner.copy_f32_range(&input, 0, &mut foreign_output, 0, 16));
        rejected!(owner.copy_f32_into_slot(&foreign_input, &mut output, 0));
        rejected!(owner.copy_f32_into_slot(&input, &mut foreign_output, 0));
        rejected!(owner.copy_f32_within(&mut foreign_output, 0, 8, 8));
        rejected!(owner.zero_f32_buffer_in_place(&mut foreign_output));
        rejected!(owner.zero_f32_range(&mut foreign_output, 0, 16));
        rejected!(owner.overwrite_f32_buffer(&values, &mut foreign_output));
        rejected!(owner.overwrite_f32_range(&values, &mut foreign_output, 0));
        rejected!(owner.download_f32_buffer(&foreign_input));
        rejected!(owner.download_f32_range(&foreign_input, 0, 16));
        rejected!(owner.clone_f32_buffer(&foreign_input));
        // Empty operations must not bypass ownership validation.
        rejected!(owner.copy_f32_range(&foreign_input, 0, &mut output, 0, 0));
        rejected!(owner.zero_f32_range(&mut foreign_output, 0, 0));
        rejected!(owner.overwrite_f32_range(&[], &mut foreign_output, 0));
        rejected!(owner.download_f32_range(&foreign_input, 0, 0));

        rejected!(owner.rms_norm_from_device(&foreign_input, &weight, 1e-5));
        rejected!(owner.rms_norm_from_device(&input, &foreign_weight, 1e-5));
        rejected!(owner.rms_norm_from_device_into(&foreign_input, &weight, 1e-5, &mut output));
        rejected!(owner.rms_norm_from_device_into(&input, &foreign_weight, 1e-5, &mut output));
        rejected!(owner.rms_norm_from_device_into(&input, &weight, 1e-5, &mut foreign_output));
        rejected!(owner.rms_norm_rows_from_device(&foreign_input, 1, &weight, 1e-5));
        rejected!(owner.rms_norm_rows_from_device_into(
            &foreign_input,
            1,
            &weight,
            1e-5,
            &mut output
        ));
        rejected!(owner.rms_norm_rows_from_device_into(
            &input,
            1,
            &foreign_weight,
            1e-5,
            &mut output
        ));
        rejected!(owner.rms_norm_rows_from_device_into(
            &input,
            1,
            &weight,
            1e-5,
            &mut foreign_output
        ));
        rejected!(owner.rms_norm_heads_from_device(&foreign_input, 1, 16, 1e-5));
        rejected!(owner.rms_norm_heads_from_device_into(&foreign_input, 1, 16, 1e-5, &mut output));
        rejected!(owner.rms_norm_heads_from_device_into(&input, 1, 16, 1e-5, &mut foreign_output));

        rejected!(
            foreign_output
                .as_device_buffer()
                .copy_from_host(&stream, &values)
        );
        rejected!(
            output
                .as_device_buffer()
                .copy_from_host(&foreign_stream, &values)
        );
        let mut host = sentinel.clone();
        rejected!(
            foreign_input
                .as_device_buffer()
                .copy_to_host(&stream, &mut host)
        );
        assert_eq!(host, sentinel);
        rejected!(
            input
                .as_device_buffer()
                .copy_to_host(&foreign_stream, &mut host)
        );
        assert_eq!(host, sentinel);

        // Same owner with distinct streams and shared allocation views is legal.
        let sibling_stream = stream.context().new_stream()?;
        let slice = input.as_device_buffer().slice(2, 4)?;
        let replacement = [9.0f32; 4];
        slice.copy_from_host(&sibling_stream, &replacement)?;
        sibling_stream.synchronize()?;
        assert_eq!(slice.to_host_vec(&stream)?, replacement);
        owner.rms_norm_from_device_into(&input, &weight, 1e-5, &mut output)?;
        let owner_sync = synchronize_owner(&owner);
        let foreign_sync = synchronize_owner(&foreign);
        assert!(
            owner_sync.is_ok() && foreign_sync.is_ok(),
            "owner={owner_sync:?}; foreign={foreign_sync:?}"
        );
        println!(
            "foreign ordinal={foreign_ordinal}: rejected mismatched owners without modifying targets"
        );
    }
    Ok(())
}
