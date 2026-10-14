use super::*;
use std::mem::size_of;

#[test]
fn decoder_endpoint_is_a_compact_exclusive_owner_handle() {
    assert!(size_of::<DecoderEndpoint>() <= 2 * size_of::<usize>());
    let sizes = [
        size_of::<DecoderChild>(),
        size_of::<ExpertChild>(),
        size_of::<crate::parallel::expert_process::NumericExpertChild>(),
    ];
    // One control block, not a copy of its weights/journal. Keep the extra startup
    // allocation below 4 KiB within the existing per-child metadata allowance.
    assert!(sizes.iter().all(|&bytes| bytes <= 4096));
    eprintln!(
        "decoder endpoint bytes={}, owner control blocks={sizes:?}",
        size_of::<DecoderEndpoint>()
    );
}

fn boot() -> super::super::decoder::DecoderBoot {
    use super::super::decoder::*;
    DecoderBoot {
        version: DECODER_WIRE_VERSION,
        rank: 7,
        checkpoint: "not-opened".into(),
        recipe: DecoderRecipeKind::Synthetic,
        segment: SegmentFrame {
            total_layers: 2,
            layers: 0..2,
            embedding: true,
            output: true,
        },
        precision: DecoderPrecision::F32,
        device: DecoderDevice::Cpu,
        kv: KvConfigFrame {
            page_size: 2,
            max_pages: 4,
            physical_pages: 4,
            max_positions: 16,
            max_batch_tokens: 4,
            session_capacity: 1,
            max_parameter_bytes: 4096,
            max_ack_polls: 8,
        },
        experts: None,
        hybrid: None,
    }
}

fn envelope(boot: &super::super::decoder::DecoderBoot, rank: u32) -> ProcessBoot {
    use crate::parallel::process::*;
    ProcessBoot {
        identity: ProcessIdentity::new(
            ProcessGroupEpoch::new(1).unwrap(),
            ferrule_common::ParallelRankId::new(rank),
            ProcessOwnerInstanceId::new(1).unwrap(),
        ),
        limits: ProcessFrameLimits::default(),
        config: serde_json::to_value(boot).unwrap(),
    }
}

#[test]
fn decoder_application_binder_runs_only_after_checked_boot_and_rebound_geometry_precedes_payload() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let resources =
        crate::parallel::expert::numeric_test_fixture::bind(file.path(), 4, false, true);
    file.close().unwrap();
    let mut boot = boot();
    let called = std::cell::Cell::new(false);
    let bad_identity = envelope(&boot, 8);
    let failure = DecoderChild::initialize_builtin(
        bad_identity,
        ProcessLaunch::new("must-not-spawn"),
        |_| {
            called.set(true);
            Ok(resources.clone())
        },
    )
    .err()
    .unwrap();
    assert!(!called.get());
    assert!(failure.message.contains("owner/rank mismatch"));
    boot.segment.total_layers = 3;
    boot.segment.layers = 0..3;
    let failure = DecoderChild::initialize_builtin(
        envelope(&boot, 7),
        ProcessLaunch::new("must-not-spawn"),
        |_| {
            called.set(true);
            Ok(resources)
        },
    )
    .err()
    .unwrap();
    assert!(called.get());
    assert!(
        failure.message.contains("layer count mismatch"),
        "{failure:?}"
    );
    assert_eq!(
        failure.quiescence,
        crate::parallel::process::ProcessQuiescence::Fenced
    );
}

#[test]
fn injected_expert_binding_is_geometry_checked_before_payload() {
    use super::super::decoder::{DecoderDevice, ExpertBoot, ExpertPlacementFrame};
    use crate::parallel::process::*;
    let file = tempfile::NamedTempFile::new().unwrap();
    let resources =
        crate::parallel::expert::numeric_test_fixture::bind(file.path(), 4, false, true);
    file.close().unwrap();
    let mut boot = boot();
    boot.segment.total_layers = 3;
    boot.segment.layers = 0..3;
    boot.experts = Some(ExpertPlacementFrame {
        source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
        source: 7,
        members: vec![109],
        entries: (0..3)
            .flat_map(|l| (0..4).map(move |e| (l, e, 109)))
            .collect(),
        max_tokens: 4,
        max_bytes: 128,
        devices: vec![DecoderDevice::Cpu],
        timeout_ms: 1000,
    });
    let mut request = envelope(&boot, 109);
    request.config = serde_json::to_value(ExpertBoot {
        expert_boot: boot,
        owner: 109,
    })
    .unwrap();
    let failure = ExpertChild::initialize(request, |_| Ok(resources))
        .err()
        .unwrap();
    assert!(
        failure.message.contains("layer count mismatch"),
        "{failure:?}"
    );
    assert_eq!(failure.quiescence, ProcessQuiescence::Fenced);
}
