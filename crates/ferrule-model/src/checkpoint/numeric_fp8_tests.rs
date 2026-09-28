//! Test-only constructor spy and publication-race checks. No production counters.
use super::*;
use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    static AFTER_VALIDATION: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}
pub(crate) fn counts() -> (usize, usize) {
    COUNTS.get()
}
pub(super) fn validated(payload: &ImmutableValidatedNumericFp8Payload) {
    COUNTS.set((counts().0 + 1, counts().1 + payload.storage_bytes()));
    let hook = AFTER_VALIDATION.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

pub(crate) struct Fixture {
    pub directory: PathBuf,
    pub source: NumericFp8Source,
    pub weight: Vec<u8>,
    pub scale: Vec<u8>,
}
impl Fixture {
    pub fn new(n: usize, k: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "numeric-proof-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let weight = vec![0x38; n * k];
        let scale: Vec<_> = (0..n.div_ceil(128) * k.div_ceil(128))
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect();
        let w = CheckpointTensorSlice {
            name: "projection.weight".into(),
            role: crate::TensorRole::AttentionQuery,
            path: directory.join("weight"),
            offset: 0,
            bytes: weight.len() as u64,
            dtype: CheckpointDType::F8E4M3,
            shape: vec![n, k],
        };
        let s = CheckpointTensorSlice {
            name: "projection.weight_scale_inv".into(),
            role: w.role.clone(),
            path: directory.join("scale"),
            offset: 0,
            bytes: scale.len() as u64,
            dtype: CheckpointDType::F32,
            shape: vec![n.div_ceil(128), k.div_ceil(128)],
        };
        std::fs::write(&w.path, &weight).unwrap();
        std::fs::write(&s.path, &scale).unwrap();
        let source = NumericFp8Source::new(
            w.clone(),
            s.clone(),
            NumericFp8Encoding::E4M3FnBlock128F32,
            CheckpointSourceFileIdentity::capture(&w.path).unwrap(),
            CheckpointSourceFileIdentity::capture(&s.path).unwrap(),
        )
        .unwrap();
        Self {
            directory,
            source,
            weight,
            scale,
        }
    }
    pub fn plan(&self, reader: &CheckpointTensorReader) -> NumericFp8Read {
        let [n, k] = self.source.matrix_shape();
        self.source.plan_tile(reader, None, 0..n, 0..k).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}

pub(crate) fn bind(source: &NumericFp8Source, expert: bool) -> crate::transformer::BoundStateDict {
    use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec};
    use crate::transformer::{ExactNameMapper, NameMapping, StateDictBinder, StateDictSchema};
    let path = ModulePath::new("projection").unwrap();
    let spec = ParameterSpec::new(
        ParameterId::new(1),
        path.clone(),
        ParameterDType::F8E4M3,
        source.weight().shape.clone(),
        if expert {
            ParameterResidency::expert(0, 0)
        } else {
            ParameterResidency::Static
        },
    )
    .unwrap()
    .with_required_scale(
        match source.encoding() {
            NumericFp8Encoding::E4M3FnBlock128Bf16 => ParameterDType::Bf16,
            NumericFp8Encoding::E4M3FnBlock128F32 => ParameterDType::F32,
        },
        source.scale().shape.clone(),
    )
    .unwrap();
    let mut schema = StateDictSchema::builder();
    schema
        .register_with_role(spec, source.weight().role.clone())
        .unwrap();
    let mut mapper = ExactNameMapper::new();
    mapper
        .insert(&source.weight().name, NameMapping::weight(path.clone()))
        .unwrap();
    mapper
        .insert(&source.scale().name, NameMapping::scale(path))
        .unwrap();
    StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(vec![source.weight().clone(), source.scale().clone()])
        .unwrap()
}

#[test]
fn constructor_once_and_clones_share_exact_ragged_proof() {
    let f = Fixture::new(259, 259);
    let reader = CheckpointTensorReader::new(100_000);
    let plan = f
        .source
        .plan_tile(&reader, None, 127..259, 125..259)
        .unwrap();
    let before = counts();
    let artifact = plan.read(&reader).unwrap();
    assert_eq!(
        counts(),
        (before.0 + 1, before.1 + artifact.storage_bytes() as usize)
    );
    let clone = artifact.clone();
    assert_eq!(artifact, clone);
    assert_eq!(
        artifact.weight_bytes().as_ptr(),
        clone.weight_bytes().as_ptr()
    );
    assert_eq!(
        artifact.scale_bytes().as_ptr(),
        clone.scale_bytes().as_ptr()
    );
    assert_eq!(
        artifact.validated_payload().weight_bytes().as_ptr(),
        artifact.weight_bytes().as_ptr()
    );
    let l = clone.validated_payload().layout();
    assert_eq!(
        (l.n, l.k, l.row_origin, l.column_origin),
        (132, 134, 127, 125)
    );
    assert_eq!(l.scale_shape().unwrap(), plan.scale_read().local_shape());
    assert_eq!(clone.provenance().source().matrix_shape(), [259, 259]);
    assert_eq!(reader.read_counters().source_checks, 4); // two files, pre + post
    assert_eq!(counts().0, before.0 + 1);
}

#[test]
fn proof_cannot_publish_if_either_source_changes_after_constructor() {
    for external in [false, true] {
        for scale in [false, true] {
            let f = Fixture::new(2, 2);
            let reader = CheckpointTensorReader::new(64);
            let plan = f.plan(&reader);
            let path = if scale {
                f.source.scale().path.clone()
            } else {
                f.source.weight().path.clone()
            };
            let replacement = f.directory.join("replacement");
            std::fs::copy(&path, &replacement).unwrap();
            AFTER_VALIDATION.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    std::fs::rename(replacement, path).unwrap();
                }))
            });
            let before = counts().0;
            let result = if external {
                plan.materialize(f.weight.clone(), f.scale.clone())
            } else {
                plan.read(&reader)
            };
            assert_eq!(
                counts().0,
                before + 1,
                "replacement occurs after successful proof construction"
            );
            assert!(result.unwrap_err().to_string().contains("stale"));
            assert!(plan.source().validate_source_identity().is_err());
        }
    }
}

#[test]
fn stale_provenance_does_not_change_the_content_proof() {
    let f = Fixture::new(2, 2);
    let reader = CheckpointTensorReader::new(64);
    let artifact = f.plan(&reader).read(&reader).unwrap();
    let proof = artifact.validated_payload().clone();
    let replacement = f.directory.join("replacement");
    std::fs::write(&replacement, f32::NAN.to_le_bytes()).unwrap();
    std::fs::rename(replacement, &f.source.scale().path).unwrap();
    assert!(artifact.decode_f32(16).is_err());
    assert!(
        artifact
            .provenance()
            .materialize(f.weight.clone(), f.scale.clone())
            .is_err()
    );
    assert_eq!(proof.scale_bytes(), &1.0f32.to_le_bytes());
    proof
        .layout()
        .validate_payload(proof.weight_bytes(), proof.scale_bytes())
        .unwrap();
}

#[inline(never)]
fn legacy_scan(weight: &[u8], scales: &[u8], encoding: NumericFp8Encoding) {
    assert!(!weight.iter().any(|byte| byte & 0x7f == 0x7f));
    for bytes in scales.chunks_exact(encoding.scale_element_bytes()) {
        let value = encoding.scale(bytes);
        assert!(value.is_finite() && value > 0.0);
    }
}
fn median(mut run: impl FnMut()) -> std::time::Duration {
    for _ in 0..3 {
        run();
    }
    let mut samples: Vec<_> = (0..21)
        .map(|_| {
            let start = std::time::Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    samples.sort();
    samples[10]
}

#[test]
#[ignore = "CPU-only release benchmark: same 32 MiB bytes and source checks, no GPU"]
fn model_proof_32_mib_validation_benchmark() {
    use std::hint::black_box;
    let f = Fixture::new(4096, 8192);
    let reader = CheckpointTensorReader::new(40 << 20);
    let plan = f.plan(&reader);
    fn measure(f: &Fixture, mut run: impl FnMut(Vec<u8>, Vec<u8>)) -> std::time::Duration {
        let mut samples = Vec::new();
        for i in 0..24 {
            // Both consumers start with the same owned read buffers. Preparing
            // repeated benchmark inputs is outside BOTH timers, not model work.
            let weight = f.weight.clone();
            let scales = f.scale.clone();
            let start = std::time::Instant::now();
            run(weight, scales);
            if i >= 3 {
                samples.push(start.elapsed());
            }
        }
        samples.sort();
        samples[10]
    }
    let old = measure(&f, |weight, scales| {
        plan.source.validate_source_identity().unwrap();
        legacy_scan(black_box(&weight), black_box(&scales), f.source.encoding());
        plan.source.validate_source_identity().unwrap();
        legacy_scan(black_box(&weight), black_box(&scales), f.source.encoding());
    });
    let before = counts();
    let new = measure(&f, |weight, scales| {
        // Vec -> Arc conversion and both source checks stay inside this timer.
        let artifact = plan.materialize(weight, scales).unwrap();
        let p = black_box(artifact.validated_payload());
        p.layout()
            .validate_lengths(p.weight_bytes().len(), p.scale_bytes().len())
            .unwrap();
        black_box(artifact);
    });
    assert_eq!(counts().0 - before.0, 24);
    eprintln!(
        "model 32 MiB from owned buffers, same source checks: old double scan={old:?}, materialize+proof reuse (includes Vec->Arc)={new:?}; reduction={:.1}%; content validations old/new=48/24, bytes old/new={}/{}",
        100.0 * (1.0 - new.as_secs_f64() / old.as_secs_f64()),
        48 * (f.weight.len() + f.scale.len()),
        counts().1 - before.1
    );
}

#[test]
#[ignore = "CPU-only one NAS expert; set FERRULE_NUMERIC_FP8_MODEL_DIR; no full model"]
fn nas_same_prepared_expert_validation_benchmark() {
    use crate::checkpoint::HfSafetensorsIndex;
    use crate::transformer::StateDictMaterializer;
    use std::io::Read;
    let directory =
        PathBuf::from(std::env::var("FERRULE_NUMERIC_FP8_MODEL_DIR").expect("set NAS path"));
    let index = HfSafetensorsIndex::open(directory.join("model.safetensors.index.json")).unwrap();
    let (name, shard) = index
        .weight_map
        .iter()
        .filter(|(name, shard)| {
            name.contains(".mlp.experts.")
                && name.ends_with(".down_proj.weight")
                && index.weight_map.get(&format!("{name}_scale_inv")) == Some(shard)
        })
        .min_by_key(|(name, _)| *name)
        .expect("one numeric expert pair");
    let path = directory.join(shard);
    let mut file = std::fs::File::open(&path).unwrap();
    let mut length = [0; 8];
    file.read_exact(&mut length).unwrap();
    let length = u64::from_le_bytes(length);
    assert!(length < 100 << 20);
    let mut header = vec![0; length as usize];
    file.read_exact(&mut header).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    let tensor = |name: &str| {
        let t = &header[name];
        let start = t["data_offsets"][0].as_u64().unwrap();
        CheckpointTensorSlice {
            name: name.into(),
            path: path.clone(),
            role: crate::TensorRole::RoutedExpertDown,
            offset: length + 8 + start,
            bytes: t["data_offsets"][1].as_u64().unwrap() - start,
            dtype: CheckpointDType::from_safetensors_dtype(t["dtype"].as_str().unwrap()),
            shape: serde_json::from_value(t["shape"].clone()).unwrap(),
        }
    };
    let w = tensor(name);
    let s = tensor(&format!("{name}_scale_inv"));
    let encoding = match s.dtype {
        CheckpointDType::Bf16 => NumericFp8Encoding::E4M3FnBlock128Bf16,
        CheckpointDType::F32 => NumericFp8Encoding::E4M3FnBlock128F32,
        _ => panic!("numeric scale"),
    };
    let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
    let source = NumericFp8Source::new(w, s, encoding, snapshot.clone(), snapshot).unwrap();
    let total = source.weight().bytes + source.scale().bytes;
    assert!(total <= 4 << 20, "only one small expert projection");
    let bound = bind(&source, true);
    let parameter = &bound.parameters()[0];
    let materializer = StateDictMaterializer::new(total).unwrap();
    let load = || {
        materializer
            .prepared_linear(parameter, parameter.role().clone())
            .unwrap()
    };
    let prepared = load();
    let artifact = prepared.numeric_fp8().unwrap();
    let before = counts();
    let old = median(|| {
        legacy_scan(
            std::hint::black_box(artifact.weight_bytes()),
            artifact.scale_bytes(),
            encoding,
        );
        legacy_scan(
            std::hint::black_box(artifact.weight_bytes()),
            artifact.scale_bytes(),
            encoding,
        );
    });
    let reuse = median(|| {
        let p = std::hint::black_box(artifact.validated_payload());
        p.layout()
            .validate_lengths(p.weight_bytes().len(), p.scale_bytes().len())
            .unwrap();
    });
    assert_eq!(
        counts(),
        before,
        "prepared proof reuse never reconstructs payload"
    );
    let before = counts();
    let reload = median(|| {
        let next = load();
        assert_eq!(
            next.numeric_fp8().unwrap().weight_bytes(),
            artifact.weight_bytes()
        );
        assert_eq!(
            next.numeric_fp8().unwrap().scale_bytes(),
            artifact.scale_bytes()
        );
        std::hint::black_box(next);
    });
    assert_eq!(counts().0 - before.0, 24);
    assert_eq!(counts().1 - before.1, 24 * total as usize);
    eprintln!(
        "NAS same prepared expert {name}, {total} bytes: legacy two content scans={old:?}, prepped proof structure={reuse:?}, verified prepared reload including IO={reload:?}; reload validations=24/{}, prepared reuse validations=0 (old baseline=48); bytes/math unchanged",
        counts().1 - before.1
    );
}
