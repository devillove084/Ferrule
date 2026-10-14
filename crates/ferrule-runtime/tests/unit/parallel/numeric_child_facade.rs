use super::*;
use std::cell::Cell;

fn envelope(boot: &NumericBoot) -> ProcessBoot {
    ProcessBoot {
        identity: ProcessIdentity::new(
            ProcessGroupEpoch::new(7).unwrap(),
            ParallelRankId::new(boot.owner),
            ProcessOwnerInstanceId::new(9).unwrap(),
        ),
        limits: ProcessFrameLimits::default(),
        config: serde_json::to_value(boot).unwrap(),
    }
}

#[test]
fn checked_boot_precedes_platform_rejection_and_never_calls_invalid_binder() {
    for case in 0..7 {
        let mut boot = codec_tests::boot();
        let mut request = envelope(&boot);
        match case {
            0 => boot.numeric_expert_boot = 0,
            1 => request.identity.rank = ParallelRankId::new(42),
            2 => request.limits.max_command_bytes = 1,
            3 => boot.placement.source = boot.owner,
            4 => boot.expected.reverse(),
            5 => boot.workspace_bytes = boot.device_bytes,
            6 => {}
            _ => unreachable!(),
        }
        request.config = serde_json::to_value(boot).unwrap();
        if case == 6 {
            request.config["numeric_profile"] = serde_json::json!("bf16");
        }
        let called = Cell::new(false);
        let failure = NumericExpertChild::initialize(request, |_| {
            called.set(true);
            panic!("invalid Boot must never call the application binder");
        })
        .err()
        .expect("invalid metadata must fail");
        assert!(!called.get(), "case {case}");
        assert_eq!(failure.kind, ProcessFailureKind::Startup);
        assert_eq!(failure.quiescence, ProcessQuiescence::Fenced);
        assert!(
            !failure.message.contains("requires CUDA"),
            "case {case}: {failure:?}"
        );
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cpu_facade_rejects_valid_metadata_without_binder_or_checkpoint_io() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = directory.path().join("checkpoint-must-not-be-opened");
    let mut boot = codec_tests::boot();
    boot.checkpoint = checkpoint.clone();
    let called = Cell::new(false);
    let failure = NumericExpertChild::initialize(envelope(&boot), |_| {
        called.set(true);
        panic!("CPU facade must not bind or read the checkpoint");
    })
    .err()
    .expect("CPU numeric execution must fail closed");
    assert!(!called.get());
    assert!(!checkpoint.exists());
    assert_eq!(failure.kind, ProcessFailureKind::Startup);
    assert_eq!(failure.quiescence, ProcessQuiescence::Fenced);
    assert_eq!(
        failure.message,
        "numeric expert requires CUDA; no CPU fallback"
    );
}

#[test]
fn numeric_stats_wire_shape_is_identical_on_cpu_and_cuda_builds() {
    let stats = NumericProcessStats {
        pid: 123,
        owner: 109,
        slot: 0,
        device: DecoderDevice::Cuda { ordinal: 1 },
        owned_experts: vec![(1, 0), (2, 0)],
        calls: 9,
        tokens: 81,
        host_payload_bytes: 256,
        host_read_bytes: 256,
        host_hits: 3,
        resident_experts: 2,
        resident_bytes: 256,
        uploads: 2,
        evictions: 0,
        pending_upload_bytes: 0,
        quarantined: false,
    };
    let json = serde_json::to_value(&stats).unwrap();
    let keys = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        keys,
        [
            "pid",
            "owner",
            "slot",
            "device",
            "owned_experts",
            "calls",
            "tokens",
            "host_payload_bytes",
            "host_read_bytes",
            "host_hits",
            "resident_experts",
            "resident_bytes",
            "uploads",
            "evictions",
            "pending_upload_bytes",
            "quarantined",
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(
        serde_json::from_value::<NumericProcessStats>(json.clone()).unwrap(),
        stats
    );
    for key in keys {
        let mut missing = json.clone();
        missing.as_object_mut().unwrap().remove(key);
        assert!(
            serde_json::from_value::<NumericProcessStats>(missing).is_err(),
            "missing {key}"
        );
    }
    let reply = serde_json::to_value(NumericReply::Stats(stats)).unwrap();
    assert_eq!(reply["type"], "stats");
}

#[test]
fn numeric_frame_preflight_is_available_through_the_narrow_process_api() {
    let size = crate::parallel::expert_process::NumericFrameSize::new(2048, 256).unwrap();
    assert_eq!(size.payload_bytes(), 5_865_728);
    assert_eq!(size.frame_bytes(), 5_866_752);
    assert!(!size.fits(ProcessFrameLimits::default()));
    let limits = ProcessFrameLimits {
        max_command_bytes: size.payload_bytes(),
        max_output_bytes: size.payload_bytes(),
        ..Default::default()
    };
    limits.validate().unwrap();
    assert!(size.fits(limits));
    assert_eq!(
        crate::parallel::expert_process::NumericFrameSize::new(usize::MAX, 1),
        Err(crate::parallel::expert_process::FrameSizeOverflow::Payload)
    );
}
