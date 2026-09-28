//! Sparse metadata only: never reads the 35B payload or initializes CUDA.
use super::*;
use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use crate::models::qwen35::{
    Qwen35Config, Qwen35HfNameMapper, Qwen35Recipe, Qwen35TensorPartitionKind,
};
use crate::transformer::{BoundDecoderResources, StateDictBinder};
const G: u64 = 1 << 30;

struct Fixture {
    path: PathBuf,
    resources: BoundDecoderResources,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "host-budget-metadata-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let config = Qwen35Config::from_value(
            &serde_json::from_str(include_str!("../../../tests/qwen35_35b_fp8_config.json"))
                .unwrap(),
        )
        .unwrap();
        let schema = Qwen35Recipe::schema(&config).unwrap();
        let mapper = Qwen35HfNameMapper::new(&config);
        let mut offset = 0;
        let slices = mapper
            .tensors()
            .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
            .map(|t| {
                let slice = CheckpointTensorSlice {
                    name: t.external_name.clone(),
                    role: TensorRole::Unknown,
                    path: path.clone(),
                    offset,
                    bytes: t.bytes(),
                    dtype: CheckpointDType::from_safetensors_dtype(t.dtype.as_str()),
                    shape: t.shape.clone(),
                };
                offset += slice.bytes;
                slice
            })
            .collect::<Vec<_>>();
        // Extents are sparse; only the inode has this length. No payload write/read.
        std::fs::File::create_new(&path)
            .unwrap()
            .set_len(offset)
            .unwrap();
        let bound = StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap();
        let resources =
            BoundDecoderResources::new(Qwen35Recipe::spec(&config).unwrap(), Arc::new(bound))
                .unwrap();
        Self { path, resources }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).unwrap();
    }
}
fn snapshot(gib: u64, used: u64) -> HostMemorySnapshot {
    HostMemorySnapshot {
        system_available_bytes: 100 * G,
        process_rss_bytes: used,
        cgroup_limit_bytes: Some(gib * G),
        cgroup_current_bytes: Some(used),
    }
}
fn typed(e: &Error) -> &HostExpertCacheError {
    match e {
        Error::ModelSource { source } => source.downcast_ref().unwrap(),
        _ => panic!("untyped: {e}"),
    }
}

#[test]
fn bound_35b_base_ledger_and_real_incremental_boundaries() {
    let f = Fixture::new();
    let base = StandardHostStartupMemory::for_resources(&f.resources, 1024, G).unwrap();
    assert_eq!(base.compressed_bytes, 1_405_263_360);
    let embedding = 248_320u64 * 2048 * 2;
    // Two 970 MiB original BF16 weights AND their LinearWeight clones remain.
    assert!(base.dense_source_bytes >= 2 * embedding);
    assert!(base.native_linear_clone_bytes >= 2 * embedding);
    assert_eq!(base.conversion_temporary_bytes, embedding * 4); // values F32 + bytes
    assert_eq!(base.reader_temporary_bytes, embedding * 2);
    assert!(base.decoded_auxiliary_bytes > 0);
    assert_eq!(base.rotary_bytes, 10 * 1024 * 64 * 4);
    assert!(base.metadata_bytes > 30_720 * 4096);
    let options = HostExpertCacheOptions {
        memory_budget: base.memory_budget().unwrap(),
        ..Default::default()
    };
    let full = HostExpertWarmPlan::new(f.resources.state_dict(), options, G).unwrap();
    assert_eq!(full.payload_bytes(), 32_216_186_880);
    let admitted = full.check_memory_snapshot(snapshot(48, G)).unwrap();
    assert_eq!(
        admitted.model_incremental_peak_bytes,
        full.memory_plan().retained_bytes().unwrap()
            + base.resident_bytes().unwrap()
            + base.temporary_bytes()
    );
    eprintln!("bound 35B host base ledger: {base:?}; admission: {admitted:?}");
    for gib in [32, 40] {
        assert!(matches!(
            typed(&full.check_memory_snapshot(snapshot(gib, G)).unwrap_err()),
            HostExpertCacheError::AvailableMemory { .. }
        ));
    }
    // Shrinking the explicit cache cap to its exact requirement cannot hide base
    // memory and make a dangerous near-payload budget pass admission.
    let tight = HostExpertWarmPlan::new(
        f.resources.state_dict(),
        HostExpertCacheOptions {
            max_bytes: full.required_bytes(),
            ..options
        },
        G,
    )
    .unwrap();
    assert!(tight.check_memory_snapshot(snapshot(40, G)).is_err());
    assert_eq!(
        tight
            .check_memory_snapshot(snapshot(48, G))
            .unwrap()
            .required_available_bytes,
        admitted.required_available_bytes
    );
    // The generic cache does not claim it can forecast a model: its real peak
    // fits available 39 GiB even with a configured 40 GiB cap (old false reject).
    let generic = HostExpertWarmPlan::new(
        f.resources.state_dict(),
        HostExpertCacheOptions::default(),
        G,
    )
    .unwrap();
    assert!(generic.check_memory_snapshot(snapshot(40, G)).is_ok());
    assert!(generic.check_memory_snapshot(snapshot(32, G)).is_err());
    let wider_cap = HostExpertWarmPlan::new(
        f.resources.state_dict(),
        HostExpertCacheOptions {
            max_bytes: 48 * G,
            ..options
        },
        G,
    )
    .unwrap();
    assert_eq!(
        wider_cap
            .check_memory_snapshot(snapshot(48, G))
            .unwrap()
            .required_available_bytes,
        admitted.required_available_bytes
    );
    assert!(StandardHostStartupMemory::for_resources(&f.resources, 1024, embedding - 1).is_err());
}

#[test]
fn worker_peaks_are_bounded_and_reused_by_serial_base_startup() {
    let f = Fixture::new();
    let base = StandardHostStartupMemory::for_resources(&f.resources, 1024, G).unwrap();
    let mut previous = None;
    for workers in [1, 4, 8] {
        let options = HostExpertCacheOptions {
            workers,
            memory_budget: base.memory_budget().unwrap(),
            ..Default::default()
        };
        let plan = HostExpertWarmPlan::new(f.resources.state_dict(), options, G).unwrap();
        let cache = plan.memory_plan();
        let largest = 2048 * 512 + (2048 / 128) * (512 / 128) * 2;
        assert_eq!(cache.worker_staging_bytes, 2 * workers as u64 * largest);
        assert_eq!(cache.worker_stack_bytes, (workers as u64) << 20);
        assert_eq!(cache.peak_bytes().unwrap(), plan.required_bytes());
        let a = plan.check_memory_snapshot(snapshot(48, G)).unwrap();
        if let Some((required, warm)) = previous {
            assert_eq!(
                required, a.required_available_bytes,
                "post-warm conversion reuses worker scratch credit"
            );
            assert!(a.warm_incremental_peak_bytes > warm);
        }
        previous = Some((a.required_available_bytes, a.warm_incremental_peak_bytes));
        // Base already physically present: cgroup current includes it, and the
        // future model phase only needs its conversion workspace.
        let reused = HostExpertCacheOptions {
            memory_budget: HostMemoryBudget {
                base_already_resident_bytes: base.resident_bytes().unwrap(),
                ..options.memory_budget
            },
            ..options
        };
        let reused = HostExpertWarmPlan::new(f.resources.state_dict(), reused, G).unwrap();
        let reused = reused
            .check_memory_snapshot(snapshot(48, G + base.resident_bytes().unwrap()))
            .unwrap();
        assert_eq!(
            a.required_available_bytes - reused.required_available_bytes,
            base.resident_bytes().unwrap()
        );
    }
}

#[test]
#[cfg(target_os = "linux")]
fn startup_memory_rejects_before_read_session_or_partial_loading() {
    let f = Fixture::new();
    let options = HostExpertCacheOptions {
        memory_budget: HostMemoryBudget {
            base_temporary_bytes: 1 << 62,
            ..Default::default()
        },
        ..Default::default()
    };
    let plan = HostExpertWarmPlan::new(f.resources.state_dict(), options, G).unwrap();
    let err = plan
        .warm(|_| panic!("must reject before starting warm/read workers"))
        .unwrap_err();
    assert!(matches!(
        typed(&err),
        HostExpertCacheError::AvailableMemory { .. }
    ));
    assert!(
        !std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(|e| std::fs::read_link(e.ok()?.path()).ok())
            .any(|p| p == f.path)
    );
}
