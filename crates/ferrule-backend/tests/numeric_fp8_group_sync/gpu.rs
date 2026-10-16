//! Public API real-driver coverage. Ignored: only the GPU owner may execute it.
use super::payload;
use crate::cuda::operators::linear::{
    CudaOperators, NumericFp8LinearPlan, NumericFp8Precision, NumericFp8ScaleType,
};

#[test]
#[ignore = "requires actual sm80+ CUDA GPU; synchronous grouped numeric FP8 vs three baseline uploads, both scale types"]
fn public_group_sync_matches_three_baseline_uploads_both_scale_types() {
    let owner = CudaOperators::new_on_device(0).expect("required CUDA owner");
    for scale_type in [NumericFp8ScaleType::Bf16, NumericFp8ScaleType::F32] {
        let payloads = [0x38, 0x40, 0x7e].map(|marker| payload(scale_type, marker));
        let layouts = payloads.each_ref().map(|payload| payload.layout());
        let packed_bytes = payloads
            .iter()
            .map(|payload| payload.storage_bytes())
            .sum::<usize>();
        owner.reset_counters();
        owner.enable_capture_safe();
        assert!(
            owner
                .upload_validated_numeric_fp8_group_sync(payloads.clone())
                .is_err()
        );
        owner.disable_capture_safe();
        assert_eq!(owner.counters().device_allocation_attempts, 0);
        assert_eq!(owner.counters().host_to_device_copies, 0);
        assert_eq!(owner.counters().stream_wide_syncs, 0);

        owner.reset_counters();
        owner.failpoints().arm_allocation();
        assert!(
            owner
                .upload_validated_numeric_fp8_group_sync(payloads.clone())
                .is_err()
        );
        assert_eq!(owner.counters().host_to_device_copies, 0);
        assert_eq!(owner.counters().stream_wide_syncs, 0);

        let baseline = payloads
            .each_ref()
            .map(|payload| owner.upload_validated_numeric_fp8_linear(payload).unwrap());
        let liveness = payloads
            .each_ref()
            .map(|payload| payload.allocation_liveness());
        owner.reset_counters();
        // Ordinary facade sync rejection cannot cancel submitted-work cleanup.
        owner.failpoints().arm_stream_sync();
        let grouped = owner
            .upload_validated_numeric_fp8_group_sync(payloads)
            .unwrap();
        let counters = owner.counters();
        assert_eq!(counters.device_allocation_attempts, 6);
        assert_eq!(counters.device_allocations, 6);
        assert_eq!(counters.device_allocation_bytes, packed_bytes as u64);
        assert_eq!(counters.host_to_device_copies, 6);
        assert_eq!(counters.host_to_device_bytes, packed_bytes as u64);
        assert_eq!(counters.stream_wide_syncs, 1);
        assert_eq!(owner.stream_wide_sync_attempts(), 1);
        assert_eq!(counters.stream_wide_sync_failures, 0);
        assert_eq!(counters.artifact_uploads, 3);
        assert_eq!(counters.artifact_upload_bytes, packed_bytes as u64);
        assert_eq!(counters.kernel_launches, 0);
        assert!(
            liveness.iter().all(|payload| !payload.is_live()),
            "sources retire before synchronous return"
        );
        assert!(
            owner.sync_stream().is_err(),
            "ordinary sync failpoint is still armed"
        );
        assert_eq!(owner.counters().stream_wide_syncs, 1);

        // Reuse the exact fixture from retained-upload contracts. Identity rows
        // expose each decoded element across both intersecting 128-scale axes.
        let input = owner.upload_f32_buffer(&[1.0, 0.0, 0.0, 1.0]).unwrap();
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::F32Tf32x3,
        ] {
            for (index, value) in [1.0f32, 2.0, 448.0].into_iter().enumerate() {
                assert_eq!(grouped[index].layout(), layouts[index]);
                assert_eq!(
                    grouped[index].storage_bytes(),
                    baseline[index].storage_bytes()
                );
                let plan = NumericFp8LinearPlan::new(layouts[index], 2, 64, precision).unwrap();
                let mut workspace = owner.numeric_fp8_linear_workspace(plan).unwrap();
                let mut old_output = owner.zero_f32_buffer(4).unwrap();
                let mut new_output = owner.zero_f32_buffer(4).unwrap();
                owner
                    .numeric_fp8_linear_into(
                        &baseline[index],
                        &input,
                        &mut old_output,
                        &mut workspace,
                        plan,
                        2,
                        2,
                    )
                    .unwrap();
                owner
                    .numeric_fp8_linear_into(
                        &grouped[index],
                        &input,
                        &mut new_output,
                        &mut workspace,
                        plan,
                        2,
                        2,
                    )
                    .unwrap();
                let old_output = owner.download_f32_buffer(&old_output).unwrap();
                let new_output = owner.download_f32_buffer(&new_output).unwrap();
                assert_eq!(
                    new_output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    old_output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{scale_type:?} {precision:?} artifact {index}"
                );
                assert_eq!(new_output, [0.0, 3.0 * value, 0.0, -4.0 * value]);
                // Downloads prove completion before consumer operands are freed.
            }
        }
    }
}
