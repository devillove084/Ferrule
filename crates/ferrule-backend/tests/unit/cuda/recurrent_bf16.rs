use super::*;

#[test]
fn mixed_byte_ranges_and_overflow() {
    let bf16 = Range::new(100, 4, 2).unwrap();
    let f32 = Range::new(106, 1, 4).unwrap();
    assert!(disjoint(&[bf16], &[f32]).is_err());
    assert!(disjoint(&[], &[bf16, f32]).is_err());
    assert!(disjoint(&[bf16, bf16], &[Range::new(108, 2, 4).unwrap()]).is_ok());
    assert!(Range::new(u64::MAX - 1, 1, 2).is_err());
    assert!(Range::new(0, usize::MAX, 4).is_err());
}

#[test]
fn native_guards_reject_invalid_descriptors_without_cuda() {
    // Non-null sentinels never reach a launch: all descriptors are invalid.
    let c = Bf16ConvArgs {
        rows: 3,
        channels: 5,
        kernel_size: 4,
        input_bf16: 2,
        weight_bf16: 4,
        history_bf16: 6,
        output_bf16: 8,
        ..Default::default()
    };
    for bad in [
        Bf16ConvArgs { rows: 0, ..c },
        Bf16ConvArgs { channels: 0, ..c },
        Bf16ConvArgs {
            kernel_size: 0,
            ..c
        },
        Bf16ConvArgs { reserved: 1, ..c },
        Bf16ConvArgs {
            rows: u32::MAX,
            ..c
        },
        Bf16ConvArgs {
            kernel_size: u32::MAX,
            ..c
        },
        Bf16ConvArgs { input_bf16: 0, ..c },
        Bf16ConvArgs {
            weight_bf16: 0,
            ..c
        },
        Bf16ConvArgs {
            history_bf16: 0,
            ..c
        },
        Bf16ConvArgs {
            output_bf16: 0,
            ..c
        },
        Bf16ConvArgs { bias_bf16: 1, ..c },
        Bf16ConvArgs { input_bf16: 3, ..c },
    ] {
        assert_ne!(unsafe { ferrule_core_bf16_conv_launch(&bad) }, 0);
    }
    let d = Bf16DeltaArgs {
        rows: 3,
        key_heads: 2,
        value_heads: 4,
        key_dim: 3,
        value_dim: 5,
        qkv_bf16: 2,
        a_bf16: 4,
        b_bf16: 6,
        a_log_bf16: 8,
        dt_bias_bf16: 10,
        state_f32: 12,
        output_bf16: 14,
        ..Default::default()
    };
    for bad in [
        Bf16DeltaArgs { rows: 0, ..d },
        Bf16DeltaArgs { key_heads: 0, ..d },
        Bf16DeltaArgs {
            value_heads: 3,
            ..d
        },
        Bf16DeltaArgs { key_dim: 0, ..d },
        Bf16DeltaArgs { value_dim: 0, ..d },
        Bf16DeltaArgs { reserved: 1, ..d },
        Bf16DeltaArgs {
            rows: u32::MAX,
            ..d
        },
        Bf16DeltaArgs {
            key_dim: u32::MAX,
            ..d
        },
        Bf16DeltaArgs { qkv_bf16: 0, ..d },
        Bf16DeltaArgs { a_bf16: 0, ..d },
        Bf16DeltaArgs { b_bf16: 0, ..d },
        Bf16DeltaArgs { a_log_bf16: 0, ..d },
        Bf16DeltaArgs {
            dt_bias_bf16: 0,
            ..d
        },
        Bf16DeltaArgs {
            output_bf16: 0,
            ..d
        },
        Bf16DeltaArgs { state_f32: 0, ..d },
        Bf16DeltaArgs { state_f32: 14, ..d },
    ] {
        assert_ne!(unsafe { ferrule_core_bf16_delta_launch(&bad) }, 0);
    }
    let n = Bf16GatedNormArgs {
        rows: 3,
        width: 5,
        epsilon: 1e-6,
        input_bf16: 2,
        weight_bf16: 4,
        gate_bf16: 6,
        output_bf16: 8,
        ..Default::default()
    };
    for bad in [
        Bf16GatedNormArgs { rows: 0, ..n },
        Bf16GatedNormArgs { width: 0, ..n },
        Bf16GatedNormArgs {
            rows: u32::MAX,
            ..n
        },
        Bf16GatedNormArgs { reserved: 1, ..n },
        Bf16GatedNormArgs { epsilon: 0.0, ..n },
        Bf16GatedNormArgs {
            epsilon: f32::NAN,
            ..n
        },
        Bf16GatedNormArgs {
            epsilon: f32::INFINITY,
            ..n
        },
        Bf16GatedNormArgs { epsilon: -1.0, ..n },
        Bf16GatedNormArgs { input_bf16: 0, ..n },
        Bf16GatedNormArgs {
            weight_bf16: 0,
            ..n
        },
        Bf16GatedNormArgs { gate_bf16: 0, ..n },
        Bf16GatedNormArgs {
            output_bf16: 0,
            ..n
        },
    ] {
        assert_ne!(unsafe { ferrule_core_bf16_gated_norm_launch(&bad) }, 0);
    }
    assert_ne!(
        unsafe { ferrule_core_bf16_conv_launch(std::ptr::null()) },
        0
    );
    assert_ne!(
        unsafe { ferrule_core_bf16_delta_launch(std::ptr::null()) },
        0
    );
    assert_ne!(
        unsafe { ferrule_core_bf16_gated_norm_launch(std::ptr::null()) },
        0
    );
}

#[test]
#[ignore = "requires CUDA GPU; legitimate shared allocation views, no forged Rust references"]
fn bf16_aliases_are_rejected_before_submission() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let root = op.upload_bf16_words(&[0x3f80; 128]).unwrap();
    let view = |offset, len| {
        CudaBf16Buffer::from_device_buffer(root.as_device_buffer().slice(offset, len).unwrap())
    };
    let conv = CausalConv1dLayout {
        rows: 3,
        channels: 5,
        kernel_size: 4,
    };
    for (input, weight, bias, history, output) in [
        (0, 32, 64, 1, 80),
        (0, 32, 64, 40, 80),
        (0, 32, 64, 60, 80),
        (0, 32, 64, 80, 1),
        (0, 32, 64, 80, 40),
        (0, 32, 64, 80, 60),
        (0, 32, 64, 80, 90),
    ] {
        let input = view(input, 15);
        let weight = view(weight, 20);
        let bias = view(bias, 5);
        let mut history = view(history, 20);
        let mut output = view(output, 15);
        op.reset_counters();
        assert!(
            op.causal_depthwise_conv1d_silu_bf16_into(
                Bf16CausalConv1dBuffers {
                    input: &input,
                    weight: &weight,
                    bias: Some(&bias),
                    history: &mut history,
                    output: &mut output
                },
                conv
            )
            .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        assert_eq!(op.download_bf16_buffer(&root).unwrap(), vec![1.0; 128]);
    }
    let delta = GatedDeltaNetLayout {
        rows: 1,
        key_heads: 1,
        value_heads: 2,
        key_dim: 3,
        value_dim: 2,
    };
    for output_offset in [1, 17, 21, 25, 29] {
        let qkv = view(0, 10);
        let a = view(16, 2);
        let b = view(20, 2);
        let a_log = view(24, 2);
        let dt_bias = view(28, 2);
        let mut output = view(output_offset, 4);
        let mut state = op.upload_f32_buffer(&[0.37; 12]).unwrap();
        op.reset_counters();
        assert!(
            op.gated_delta_net_bf16_into(
                Bf16GatedDeltaNetBuffers {
                    qkv: &qkv,
                    a: &a,
                    b: &b,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    state: &mut state,
                    output: &mut output
                },
                delta
            )
            .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        assert_eq!(op.download_bf16_buffer(&root).unwrap(), vec![1.0; 128]);
        assert_eq!(op.download_f32_buffer(&state).unwrap(), vec![0.37; 12]);
    }
    for output_offset in [0, 1, 34, 60] {
        let input = view(0, 15);
        let weight = view(32, 5);
        let gate = view(64, 15);
        let mut output = view(output_offset, 15);
        op.reset_counters();
        assert!(
            op.gated_rms_norm_bf16_into(
                &input,
                &weight,
                &gate,
                &mut output,
                F32RowsLayout { rows: 3, width: 5 },
                1e-6
            )
            .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        assert_eq!(op.download_bf16_buffer(&root).unwrap(), vec![1.0; 128]);
    }
    // Read/read overlap is legal; writers remain disjoint.
    let input = view(0, 15);
    let weight = view(0, 5);
    let gate = view(0, 15);
    let mut output = view(15, 15);
    op.reset_counters();
    op.gated_rms_norm_bf16_into(
        &input,
        &weight,
        &gate,
        &mut output,
        F32RowsLayout { rows: 3, width: 5 },
        1e-6,
    )
    .unwrap();
    assert_eq!(op.counters().kernel_launches, 1);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    op.download_bf16_buffer(&output).unwrap();
}

#[test]
fn legacy_and_plan_storage_paths_are_the_same_type() {
    let plan = crate::cuda_plan::Bf16RecurrentStorage {
        conv_history_bytes: 256,
        recurrent_state_bytes: 240,
    };
    let canonical: Bf16RecurrentStorage = plan;
    let legacy: crate::cuda::context::recurrent_bf16::Bf16RecurrentStorage = canonical;
    assert_eq!(legacy, plan);
    let rows = crate::cuda_plan::F32RowsLayout { rows: 3, width: 5 };
    let _: crate::cuda::operators::recurrent::F32RowsLayout = rows;
}
