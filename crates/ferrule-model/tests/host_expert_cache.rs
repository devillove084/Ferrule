//! CPU-only compressed residency/admission and real NAS probe. No CUDA required.
#![cfg(target_os = "linux")]
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::nn::{
    ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::transformer::host_experts::{
    HostExpertCacheError, HostExpertCacheOptions, HostExpertWarmPlan,
};
use ferrule_model::transformer::{
    BoundStateDict, ExactNameMapper, NameMapping, StateDictBinder, StateDictMaterializer,
    StateDictSchema,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture {
    directory: PathBuf,
    bound: BoundStateDict,
}
impl Fixture {
    fn new(experts: usize, invalid_payload: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "host-experts-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let file = directory.join("shard");
        let mut data = Vec::new();
        let mut slices = Vec::new();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        for expert in 0..experts {
            for (projection, role) in [
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
            ]
            .into_iter()
            .enumerate()
            {
                let name = format!("layer.0.expert.{expert}.projection.{projection}");
                let path = ModulePath::new(&name).unwrap();
                let spec = ParameterSpec::new(
                    ParameterId::new((expert * 3 + projection + 1) as u64),
                    path.clone(),
                    ParameterDType::F8E4M3,
                    vec![2, 2],
                    ParameterResidency::expert(0, expert),
                )
                .unwrap()
                .with_required_scale(ParameterDType::Bf16, vec![1, 1])
                .unwrap();
                schema.register_with_role(spec, role.clone()).unwrap();
                mapper
                    .insert(format!("{name}.weight"), NameMapping::weight(path.clone()))
                    .unwrap();
                mapper
                    .insert(format!("{name}.scale"), NameMapping::scale(path))
                    .unwrap();
                slices.push(CheckpointTensorSlice {
                    name: format!("{name}.weight"),
                    role: role.clone(),
                    path: file.clone(),
                    offset: data.len() as u64,
                    bytes: 4,
                    dtype: CheckpointDType::F8E4M3,
                    shape: vec![2, 2],
                });
                data.extend([if invalid_payload { 0x7f } else { 0x38 }; 4]);
                slices.push(CheckpointTensorSlice {
                    name: format!("{name}.scale"),
                    role,
                    path: file.clone(),
                    offset: data.len() as u64,
                    bytes: 2,
                    dtype: CheckpointDType::Bf16,
                    shape: vec![1, 1],
                });
                data.extend(half::bf16::ONE.to_bits().to_le_bytes());
            }
        }
        std::fs::write(&file, data).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        Self { directory, bound }
    }
    fn plan(&self) -> HostExpertWarmPlan {
        HostExpertWarmPlan::new(
            &self.bound,
            HostExpertCacheOptions {
                max_bytes: 8 << 20,
                ..Default::default()
            },
            64,
        )
        .unwrap()
    }
    fn open_fds(&self) -> usize {
        std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(|e| std::fs::read_link(e.ok()?.path()).ok())
            .filter(|p| p.starts_with(&self.directory))
            .count()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}
fn typed(error: &ferrule_common::Error) -> &HostExpertCacheError {
    match error {
        ferrule_common::Error::ModelSource { source } => source.downcast_ref().unwrap(),
        _ => panic!("expected typed host error: {error}"),
    }
}

#[test]
fn metadata_only_budget_and_actual_memory_admission() {
    let fixture = Fixture::new(3, false);
    let plan = fixture.plan();
    assert_eq!(plan.payload_bytes(), 54);
    assert_eq!(plan.experts(), 3);
    assert_eq!(plan.source_count(), 1);
    let error = plan.check_available_bytes(1).unwrap_err();
    assert!(matches!(
        typed(&error),
        HostExpertCacheError::AvailableMemory { .. }
    ));
    for limits in [
        HostExpertCacheOptions {
            max_experts: 2,
            ..Default::default()
        },
        HostExpertCacheOptions {
            max_bytes: 54,
            ..Default::default()
        },
    ] {
        let error = HostExpertWarmPlan::new(&fixture.bound, limits, 64).unwrap_err();
        assert!(matches!(typed(&error), HostExpertCacheError::Budget { .. }));
    }
    // Planning depends only on captured metadata, even if the source disappears.
    std::fs::remove_file(fixture.directory.join("shard")).unwrap();
    assert_eq!(fixture.plan().payload_bytes(), 54);
    assert!(fixture.plan().warm(|_| {}).is_err());
}

#[test]
fn warm_cap_hits_share_exact_proof_zero_rereads_and_image_isolation() {
    let fixture = Fixture::new(5, false);
    let required = fixture.plan().required_bytes();
    let plan = HostExpertWarmPlan::new(
        &fixture.bound,
        HostExpertCacheOptions {
            max_bytes: required,
            max_experts: 5,
            ..Default::default()
        },
        64,
    )
    .unwrap();
    let cache = Arc::new(plan.warm(|_| {}).unwrap());
    let stats = cache.stats();
    assert_eq!(stats.payload_bytes, 90);
    assert_eq!(stats.reserved_bytes, stats.budget_bytes);
    assert_eq!(stats.io.bytes_read, 90);
    assert_eq!(stats.io.open_calls, 1);
    assert_eq!(stats.io.source_checks, 2); // once per unique source, pre + post
    assert_eq!(fixture.open_fds(), 0);
    let materializer = StateDictMaterializer::new(64).unwrap();
    materializer.attach_host_experts(&cache).unwrap();
    cache.preflight().unwrap();
    for binding in fixture.bound.parameters() {
        let ParameterResidency::Expert { layer, expert } = *binding.residency() else {
            unreachable!()
        };
        let first = materializer
            .expert_parameter(layer, expert, binding)
            .unwrap();
        let second = materializer
            .expert_parameter(layer, expert, binding)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            first.numeric_fp8().unwrap().weight_bytes().as_ptr(),
            second
                .numeric_fp8()
                .unwrap()
                .validated_payload()
                .weight_bytes()
                .as_ptr()
        );
    }
    assert_eq!(cache.hits(), 30);
    assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
    assert_eq!(materializer.checkpoint_read_counters().read_calls, 0);
    // The unchanged provider metadata/payload contract accepts cached proofs,
    // without a CUDA dependency or tensor rereads.
    let bindings: [_; 3] = std::array::from_fn(|i| fixture.bound.parameters()[i].clone());
    let metadata =
        ferrule_model::transformer::ExpertMetadata::new(0, 0, bindings.clone(), None).unwrap();
    let linears = bindings
        .iter()
        .map(|binding| {
            ferrule_model::transformer::PreparedLinear::from_parameter(
                materializer.expert_parameter(0, 0, binding).unwrap(),
                binding.role().clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let payload = ferrule_model::transformer::PreparedSwiGlu::new(
        linears[0].clone(),
        linears[1].clone(),
        linears[2].clone(),
        None,
    )
    .unwrap();
    metadata.validate_payload(&payload).unwrap();
    assert_eq!(materializer.checkpoint_read_counters().read_calls, 0);
    let alien = Fixture::new(5, false);
    assert!(
        materializer
            .expert_parameter(0, 0, &alien.bound.parameters()[0])
            .is_err()
    );
    let other = Arc::new(alien.plan().warm(|_| {}).unwrap());
    assert_ne!(cache.generation(), other.generation());
    assert!(materializer.attach_host_experts(&other).is_err());
    let weak = Arc::downgrade(&cache);
    drop(cache);
    assert!(weak.upgrade().is_some());
    drop(materializer);
    assert!(weak.upgrade().is_none());
}

#[test]
fn sources_changed_preflight_and_failed_warm_reclaim_handles() {
    let fixture = Fixture::new(20, false);
    let cache = fixture.plan().warm(|_| {}).unwrap();
    let path = fixture.directory.join("shard");
    let old = fixture.directory.join("old");
    std::fs::rename(&path, old).unwrap();
    std::fs::write(&path, vec![0u8; 360]).unwrap();
    assert!(matches!(
        typed(&cache.preflight().unwrap_err()),
        HostExpertCacheError::SourceChanged { .. }
    ));
    assert!(fixture.plan().warm(|_| {}).is_err());
    assert_eq!(fixture.open_fds(), 0);
    let invalid = Fixture::new(128, true);
    assert!(invalid.plan().warm(|_| {}).is_err());
    assert_eq!(invalid.open_fds(), 0);
    // Mutation after preflight/FD open, before the worker reads. Even when all
    // original bytes remain readable through the old FD, post-check must fail.
    let raced = Fixture::new(10, false);
    let mut changed = false;
    assert!(
        raced
            .plan()
            .warm(|p| {
                if p.completed_parameters == 0 && !changed {
                    changed = true;
                    let path = raced.directory.join("shard");
                    std::fs::rename(&path, raced.directory.join("old")).unwrap();
                    std::fs::write(path, vec![0u8; 180]).unwrap();
                }
            })
            .is_err()
    );
    assert_eq!(raced.open_fds(), 0);
}

fn physical_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("read_bytes:"))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

fn nfs_server_read_bytes(directory: &std::path::Path) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/mountstats").ok()?;
    let mut selected = false;
    let mut best = 0;
    let mut result = None;
    for line in text.lines() {
        if line.starts_with("device ") {
            selected = false;
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() >= 8
                && fields[7].starts_with("nfs")
                && directory.starts_with(fields[4])
                && fields[4].len() >= best
            {
                selected = true;
                best = fields[4].len();
            }
        } else if selected && let Some(bytes) = line.trim().strip_prefix("bytes:") {
            result = bytes.split_whitespace().nth(4).and_then(|s| s.parse().ok());
        }
    }
    result
}

#[test]
#[ignore = "reads all routed NAS weights on CPU; set FERRULE_NUMERIC_FP8_MODEL_DIR, no cache drops"]
fn nas_full_compressed_prewarm_then_zero_read_host_pass() {
    let directory = std::path::PathBuf::from(
        std::env::var("FERRULE_NUMERIC_FP8_MODEL_DIR").expect("set FERRULE_NUMERIC_FP8_MODEL_DIR"),
    );
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(&directory).unwrap();
    let plan = HostExpertWarmPlan::new(
        resources.state_dict(),
        HostExpertCacheOptions::default(),
        64 << 20,
    )
    .unwrap();
    assert_eq!(plan.experts(), 10240);
    assert_eq!(plan.payload_bytes(), 32_216_186_880);
    eprintln!(
        "CPU full NAS warm: experts={} sources={} payload={} admitted={} budget={}; existing OS cache is not dropped, so pass1 is NOT asserted cold",
        plan.experts(),
        plan.source_count(),
        plan.payload_bytes(),
        plan.required_bytes(),
        HostExpertCacheOptions::default().max_bytes
    );
    let physical_before = physical_read_bytes();
    let nfs_before = nfs_server_read_bytes(&directory);
    let cache = Arc::new(
        plan.warm(|p| {
            eprintln!(
                "warm {}/{} bytes {}/{} projections {:.2}s",
                p.read_bytes,
                p.total_bytes,
                p.completed_parameters,
                p.total_parameters,
                p.elapsed.as_secs_f64()
            )
        })
        .unwrap(),
    );
    let first_physical = physical_read_bytes() - physical_before;
    let nfs_after = nfs_server_read_bytes(&directory);
    eprintln!(
        "pass1 mount-wide NFS server read delta={:?} (may include other processes)",
        nfs_after.zip(nfs_before).map(|(a, b)| a.saturating_sub(b))
    );
    eprintln!(
        "CPU full NAS pass1: stats={:?} physical_read_bytes={first_physical}",
        cache.stats()
    );
    let materializer = StateDictMaterializer::new(64 << 20).unwrap();
    materializer.attach_host_experts(&cache).unwrap();
    let physical_before = physical_read_bytes();
    let nfs_before = nfs_server_read_bytes(&directory);
    let second = std::time::Instant::now();
    cache.preflight().unwrap();
    let mut bytes = 0;
    let mut checksum = 0u64;
    for binding in resources.state_dict().parameters() {
        let ParameterResidency::Expert { layer, expert } = *binding.residency() else {
            continue;
        };
        let parameter = materializer
            .expert_parameter(layer, expert, binding)
            .unwrap();
        let artifact = parameter.numeric_fp8().unwrap();
        bytes += artifact.storage_bytes();
        // Touch every host page, not just metadata, without rescanning FP8 values.
        for chunk in artifact
            .weight_bytes()
            .chunks(4096)
            .chain(artifact.scale_bytes().chunks(4096))
        {
            checksum = checksum.wrapping_add(chunk[0] as u64);
        }
    }
    let second_physical = physical_read_bytes() - physical_before;
    eprintln!(
        "pass2 mount-wide NFS server read delta={:?} (may include other processes)",
        nfs_server_read_bytes(&directory)
            .zip(nfs_before)
            .map(|(a, b)| a.saturating_sub(b))
    );
    eprintln!(
        "CPU full host pass2: elapsed={:?} bytes={bytes} hits={} physical_read_bytes={second_physical} checkpoint_io={:?} checksum={checksum}",
        second.elapsed(),
        cache.hits(),
        materializer.checkpoint_read_counters()
    );
    assert_eq!(bytes, cache.stats().payload_bytes);
    assert_eq!(cache.hits(), 30720);
    assert_eq!(materializer.checkpoint_read_counters().read_calls, 0);
    assert_eq!(second_physical, 0);
}
