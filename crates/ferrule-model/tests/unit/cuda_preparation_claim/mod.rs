//! Claim protocol tests. Host phase tests never construct CUDA; device custody
//! and missing-proof tests are real CUDA tests compiled but intentionally ignored.
use super::tests::{claim, device, schema};
use super::*;

#[test]
fn image_authority_is_single_use_and_cannot_finish_without_prepare() {
    let phase = ImagePreparation::new();
    assert!(!phase.is_prepared());
    assert!(phase.prepared().is_err());
    phase.begin().unwrap();
    assert!(!phase.is_prepared());
    assert!(phase.begin().is_err());
    phase.prepared().unwrap();
    assert!(phase.is_prepared());
    assert!(phase.begin().is_err());
    assert!(phase.prepared().is_err());
}

#[test]
#[ignore = "requires CUDA GPU; opaque admission and same-ordinal foreign owner"]
fn foreign_device_and_underadmission_cannot_use_claim() {
    let a = device();
    let b = device();
    let token = claim(&a).unwrap();
    assert!(token.begin_image(&b).is_err());
    assert!(!token.image.is_prepared());
    let required = HybridCudaMemoryEstimate {
        per_sequence_state_bytes: token.estimate.per_sequence_state_bytes,
        kv_bytes: token.estimate.kv_bytes + 1,
        weight_bytes_upper_bound: 128,
        workspace_bytes_upper_bound: 128,
    };
    assert!(token.validate_estimate(required, 1).is_err());
    assert!(token.validate_estimate(token.estimate, 2).is_err());
    assert!(claim(&a.clone()).is_err());
    assert!(
        token.finish(Ok(())).is_err(),
        "public Ok is not an image authority"
    );
    assert!(!a.inner.claimed.get());
    assert!(!a.needs_quarantine());
    drop(claim(&a).unwrap());
    drop(claim(&b).unwrap());
}

#[test]
#[ignore = "requires CUDA GPU; final completion cannot be forgiven by later fences"]
fn unknown_final_transfer_retains_success_value_claim_and_budget() {
    let device = device();
    let token = claim(&device).unwrap();
    token.begin_image(&device).unwrap();
    token.image_prepared().unwrap();
    let retirement = token.mixed_source_retirement();
    assert!(
        !retirement.is_complete(),
        "image_prepared is not final claim completion"
    );
    let value = Rc::new(());
    device.inner.reject_next_fence.set(true);
    assert!(token.finish(Ok(Rc::clone(&value))).is_err());
    assert!(
        !retirement.is_complete(),
        "Unknown must not permit source retirement"
    );
    assert_eq!(
        Rc::strong_count(&value),
        2,
        "successful owner's custody retained"
    );
    assert!(device.needs_quarantine());
    assert!(device.inner.claimed.get());
    device.operators().sync_stream().unwrap();
    device.operators().sync_upload_stream().unwrap();
    assert!(claim(&device).is_err());
}

#[test]
#[ignore = "requires CUDA GPU; failed post-image state construction uses shared rollback"]
fn late_failure_and_live_state_retain_claim_until_proven_clean() {
    let device = device();
    let token = claim(&device).unwrap();
    token.begin_image(&device).unwrap();
    token.image_prepared().unwrap();
    let mut lifecycle = CudaHybridSequenceLifecycle::new(schema(), device.clone()).unwrap();
    let state = lifecycle.create().unwrap();
    let bytes = device.live_state_bytes();
    assert!(
        token
            .finish::<()>(Err(error("late factory failure with live custody")))
            .is_err()
    );
    assert!(device.needs_quarantine());
    assert!(device.inner.claimed.get());
    assert_eq!(device.live_state_bytes(), bytes);
    assert!(lifecycle.try_release(state).is_err());
    assert!(claim(&device).is_err());
}

#[test]
#[ignore = "requires CUDA GPU; clean factory failure permits exactly a fresh token"]
fn clean_late_failure_drops_all_temporaries_before_refund() {
    let device = device();
    let token = claim(&device).unwrap();
    let retirement = token.mixed_source_retirement();
    let result: Result<()> = (|| {
        token.begin_image(&device)?;
        token.image_prepared()?;
        let mut lifecycle = CudaHybridSequenceLifecycle::new(schema(), device.clone())?;
        let _state = lifecycle.create()?;
        Err(error("injected post-image state/pool assembly failure"))
    })();
    assert!(token.finish(result).is_err());
    assert!(
        !retirement.is_complete(),
        "rollback is not successful image transfer"
    );
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
    assert!(!device.inner.claimed.get());
    drop(claim(&device).unwrap());
}

#[test]
#[ignore = "requires CUDA GPU; committed image consumes device authority permanently"]
fn successful_transfer_never_reopens_the_device_factory() {
    let device = device();
    let token = claim(&device).unwrap();
    token.begin_image(&device).unwrap();
    token.image_prepared().unwrap();
    let retirement = token.mixed_source_retirement();
    assert!(!retirement.is_complete());
    let image = token.finish(Ok(device.clone())).unwrap();
    assert!(retirement.is_complete());
    assert!(device.inner.claimed.get());
    assert!(claim(&image).is_err());
    drop(image);
    assert!(claim(&device).is_err());
}

#[test]
fn claim_estimate_checks_each_independent_admission_domain() {
    let admitted = HybridCudaMemoryEstimate {
        per_sequence_state_bytes: 128,
        kv_bytes: 256,
        weight_bytes_upper_bound: 512,
        workspace_bytes_upper_bound: 1024,
    };
    validate_claim_estimate(admitted, 4, admitted, 4).unwrap();
    assert!(validate_claim_estimate(admitted, 4, admitted, 5).is_err());
    for field in 0..4 {
        let mut required = admitted;
        match field {
            0 => required.per_sequence_state_bytes += 1,
            1 => required.kv_bytes += 1,
            2 => required.weight_bytes_upper_bound += 1,
            _ => required.workspace_bytes_upper_bound += 1,
        }
        assert!(validate_claim_estimate(admitted, 4, required, 4).is_err());
    }
}

#[test]
#[ignore = "requires CUDA GPU; unknown malformed success cannot drop caller custody"]
fn unknown_unprepared_success_also_retains_its_value() {
    let device = device();
    let token = claim(&device).unwrap();
    let value = Rc::new(());
    device.inner.reject_next_fence.set(true);
    assert!(token.finish(Ok(Rc::clone(&value))).is_err());
    assert_eq!(Rc::strong_count(&value), 2);
    assert!(device.needs_quarantine());
    assert!(device.inner.claimed.get());
}
