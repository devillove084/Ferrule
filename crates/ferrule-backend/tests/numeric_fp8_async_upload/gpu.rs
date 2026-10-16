//! Real-driver public-leaf coverage. Compile/list is not GPU qualification.
//! Execution belongs to the GPU owner, with an external timeout and one thread.
use super::payload;
use crate::cuda::operators::linear::{
    CudaNumericFp8UploadHostCharge, CudaOperators, NumericFp8LinearPlan, NumericFp8Precision,
    NumericFp8ScaleType,
};
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires actual sm80+ CUDA GPU; retained public upload vs synchronous numeric FP8, both scale types"]
fn public_stage_submit_poll_drain_matches_sync_both_scale_types() {
    let owner = CudaOperators::new_on_device(0).expect("required CUDA owner");
    // Same visible ordinal, distinct Rust owner/streams. Foreign-ticket checks
    // must reject BEFORE any driver operation; no invalid context use is needed.
    let foreign = CudaOperators::new_on_device(0).expect("second CUDA owner");
    for scale_type in [NumericFp8ScaleType::Bf16, NumericFp8ScaleType::F32] {
        let payloads = [0x38, 0x40, 0x7e].map(|marker| payload(scale_type, marker));
        let layouts = payloads.each_ref().map(|payload| payload.layout());
        let charge = CudaNumericFp8UploadHostCharge::for_payloads(&payloads).unwrap();
        let packed_bytes = payloads
            .iter()
            .map(|payload| payload.storage_bytes())
            .sum::<usize>();
        assert_eq!(charge.pageable_payload_bytes, packed_bytes);
        assert_eq!(charge.pinned_allocation_bytes, packed_bytes);
        assert_eq!(
            charge.total_bytes,
            2 * packed_bytes + charge.ticket_metadata_bytes
        );

        let before_rejection = owner.counters();
        for rejected_budget in [packed_bytes, charge.total_bytes - 1] {
            let error = owner
                .stage_numeric_fp8_upload(payloads.clone(), rejected_budget)
                .err()
                .expect("insufficient pageable + pinned charge must fail");
            assert!(error.to_string().contains("host budget exceeded"));
        }
        assert_eq!(
            owner.counters().device_allocation_attempts,
            before_rejection.device_allocation_attempts
        );

        let synchronous = payloads
            .each_ref()
            .map(|payload| owner.upload_validated_numeric_fp8_linear(payload).unwrap());
        let liveness = payloads
            .each_ref()
            .map(|payload| payload.allocation_liveness());
        let before_publish = owner.counters().artifact_uploads;
        let mut ticket = owner
            .stage_numeric_fp8_upload(payloads, charge.total_bytes)
            .unwrap();
        let generation = ticket.generation();
        assert!(generation > 0);
        assert_eq!(ticket.host_charge(), charge);
        assert!(
            owner.poll_numeric_fp8_upload(&mut ticket).is_err(),
            "staged is not ready"
        );
        assert_eq!(owner.counters().artifact_uploads, before_publish);
        assert!(liveness.iter().all(|payload| payload.is_live()));

        let before_foreign = foreign.counters();
        let error = foreign
            .submit_numeric_fp8_upload(&mut ticket)
            .err()
            .unwrap();
        assert!(error.to_string().contains("foreign or retired ticket"));
        assert!(foreign.poll_numeric_fp8_upload(&mut ticket).is_err());
        assert!(foreign.drain_numeric_fp8_upload(&mut ticket).is_err());
        assert_eq!(
            foreign.counters().device_allocation_attempts,
            before_foreign.device_allocation_attempts
        );
        assert_eq!(
            foreign.counters().host_to_device_bytes,
            before_foreign.host_to_device_bytes
        );

        owner.submit_numeric_fp8_upload(&mut ticket).unwrap();
        assert!(
            liveness.iter().all(|payload| payload.is_live()),
            "enqueue/wait is not host completion"
        );
        assert_eq!(owner.counters().artifact_uploads, before_publish);
        assert!(
            owner.submit_numeric_fp8_upload(&mut ticket).is_err(),
            "never re-record"
        );
        assert!(foreign.poll_numeric_fp8_upload(&mut ticket).is_err());
        assert!(foreign.drain_numeric_fp8_upload(&mut ticket).is_err());

        let deadline = Instant::now() + Duration::from_secs(10);
        let asynchronous = loop {
            if let Some(artifacts) = owner.poll_numeric_fp8_upload(&mut ticket).unwrap() {
                break artifacts;
            }
            assert!(liveness.iter().all(|payload| payload.is_live()));
            assert!(Instant::now() < deadline, "retained upload poll timed out");
            std::thread::yield_now();
        };
        assert!(liveness.iter().all(|payload| !payload.is_live()));
        assert_eq!(owner.counters().artifact_uploads, before_publish + 1);
        assert!(
            owner.poll_numeric_fp8_upload(&mut ticket).is_err(),
            "publish once"
        );
        assert!(owner.submit_numeric_fp8_upload(&mut ticket).is_err());
        assert_eq!(owner.counters().artifact_uploads, before_publish + 1);
        assert_eq!(
            asynchronous
                .iter()
                .map(|artifact| artifact.storage_bytes())
                .sum::<usize>(),
            packed_bytes
        );

        // The existing 2x2 fixture crosses both 128-scale boundaries. Two basis
        // input rows expose every dequantized value, not merely an aggregate sum.
        let input = owner.upload_f32_buffer(&[1.0, 0.0, 0.0, 1.0]).unwrap();
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::F32Tf32x3,
        ] {
            for (index, value) in [1.0f32, 2.0, 448.0].into_iter().enumerate() {
                assert_eq!(asynchronous[index].layout(), layouts[index]);
                assert_eq!(
                    asynchronous[index].storage_bytes(),
                    synchronous[index].storage_bytes()
                );
                let plan = NumericFp8LinearPlan::new(layouts[index], 2, 64, precision).unwrap();
                let mut workspace = owner.numeric_fp8_linear_workspace(plan).unwrap();
                let mut baseline = owner.zero_f32_buffer(4).unwrap();
                let mut uploaded = owner.zero_f32_buffer(4).unwrap();
                owner
                    .numeric_fp8_linear_into(
                        &synchronous[index],
                        &input,
                        &mut baseline,
                        &mut workspace,
                        plan,
                        2,
                        2,
                    )
                    .unwrap();
                owner
                    .numeric_fp8_linear_into(
                        &asynchronous[index],
                        &input,
                        &mut uploaded,
                        &mut workspace,
                        plan,
                        2,
                        2,
                    )
                    .unwrap();
                let baseline = owner.download_f32_buffer(&baseline).unwrap();
                let uploaded = owner.download_f32_buffer(&uploaded).unwrap();
                assert_eq!(
                    uploaded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    baseline.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{scale_type:?} {precision:?} artifact {index}"
                );
                assert_eq!(uploaded, [0.0, 3.0 * value, 0.0, -4.0 * value]);
                // Downloads prove the actual consumers finished before artifact
                // and scratch custody is released at the end of the scope.
            }
        }

        for submit_before_drain in [false, true] {
            let payloads = [0x38, 0x40, 0x7e].map(|marker| payload(scale_type, marker));
            let liveness = payloads
                .each_ref()
                .map(|payload| payload.allocation_liveness());
            let before_drain = owner.counters().artifact_uploads;
            let mut canceled = owner
                .stage_numeric_fp8_upload(payloads, charge.total_bytes)
                .unwrap();
            assert!(canceled.generation() > generation);
            if submit_before_drain {
                owner.submit_numeric_fp8_upload(&mut canceled).unwrap();
            }
            owner.drain_numeric_fp8_upload(&mut canceled).unwrap();
            assert!(liveness.iter().all(|payload| !payload.is_live()));
            assert!(owner.poll_numeric_fp8_upload(&mut canceled).is_err());
            assert!(owner.submit_numeric_fp8_upload(&mut canceled).is_err());
            assert_eq!(
                owner.counters().artifact_uploads,
                before_drain,
                "drain must not publish"
            );
        }
    }
}
