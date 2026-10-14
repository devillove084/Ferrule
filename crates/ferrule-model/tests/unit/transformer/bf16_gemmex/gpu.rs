use super::*;
use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec};
use crate::support::TensorRole;
use crate::transformer::{
    BoundStateDict, ExactNameMapper, ExpertCacheLimits, ExpertMetadata, NameMapping,
    NumericFp8Precision as P, StateDictBinder, StateDictMaterializer, StateDictSchema,
};
use ferrule_backend::cuda::operators::linear::Bf16GemmExPlan;
use std::{
    io::Write,
    path::{Path, PathBuf},
};
const MIXED: P = P::Bf16ModelMixedF32Accumulate;
#[derive(Clone)]
struct Case {
    name: String,
    m: usize,
    n: usize,
    k: usize,
    role: TensorRole,
    expert: Option<usize>,
}
fn cases(dir: &Path) -> Vec<Case> {
    std::fs::read_to_string(dir.join("cases.txt"))
        .unwrap()
        .lines()
        .map(|line| {
            let f = line.split_whitespace().collect::<Vec<_>>();
            assert_eq!(f.len(), 4);
            let (role, expert) = if let Some(e) = f[0].strip_prefix("expert") {
                let (id, kind) = e.split_once('_').unwrap();
                (
                    match kind {
                        "gate" => TensorRole::RoutedExpertGate,
                        "up" => TensorRole::RoutedExpertUp,
                        "down" => TensorRole::RoutedExpertDown,
                        _ => panic!("role"),
                    },
                    Some(id.parse().unwrap()),
                )
            } else {
                (
                    match f[0] {
                        "LinearAttentionQkv" => TensorRole::LinearAttentionQkv,
                        "LinearAttentionZ" => TensorRole::LinearAttentionZ,
                        "AttentionOutput" => TensorRole::AttentionOutput,
                        "SharedExpertGate" => TensorRole::SharedExpertGate,
                        "SharedExpertUp" => TensorRole::SharedExpertUp,
                        "SharedExpertDown" => TensorRole::SharedExpertDown,
                        _ => panic!("role"),
                    },
                    None,
                )
            };
            Case {
                name: f[0].into(),
                m: f[1].parse().unwrap(),
                n: f[2].parse().unwrap(),
                k: f[3].parse().unwrap(),
                role,
                expert,
            }
        })
        .collect()
}
fn load(dir: &Path, c: &Case, suffix: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{}.{}", c.name, suffix))).unwrap()
}
fn words(bytes: &[u8]) -> Vec<u16> {
    assert!(bytes.len().is_multiple_of(2));
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes([v[0], v[1]]))
        .collect()
}
fn layout(c: &Case) -> ferrule_backend::cuda_plan::NumericFp8Layout {
    ferrule_backend::cuda_plan::NumericFp8Layout {
        n: c.n,
        k: c.k,
        row_origin: 0,
        column_origin: 0,
        scale_type: ferrule_backend::cuda_plan::NumericFp8ScaleType::Bf16,
    }
}
fn bound(dir: &Path, cases: &[Case]) -> BoundStateDict {
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let module = ModulePath::new(&c.name).unwrap();
        let scale = vec![c.n / 128, c.k / 128];
        schema
            .register_with_role(
                ParameterSpec::new(
                    ParameterId::new(i as u64 + 1),
                    module.clone(),
                    ParameterDType::F8E4M3,
                    vec![c.n, c.k],
                    c.expert.map_or(ParameterResidency::layer(0), |e| {
                        ParameterResidency::expert(0, e)
                    }),
                )
                .unwrap()
                .with_required_scale(ParameterDType::Bf16, scale.clone())
                .unwrap(),
                c.role.clone(),
            )
            .unwrap();
        mapper
            .insert(&c.name, NameMapping::weight(module.clone()))
            .unwrap();
        mapper
            .insert(format!("{}_scale", c.name), NameMapping::scale(module))
            .unwrap();
        for (name, suffix, shape, dtype, bytes) in [
            (
                c.name.clone(),
                "weight.fp8",
                vec![c.n, c.k],
                CheckpointDType::F8E4M3,
                c.n * c.k,
            ),
            (
                format!("{}_scale", c.name),
                "scale.bf16",
                scale,
                CheckpointDType::Bf16,
                c.n * c.k / 8192,
            ),
        ] {
            slices.push(CheckpointTensorSlice {
                name,
                path: dir.join(format!("{}.{}", c.name, suffix)),
                role: c.role.clone(),
                offset: 0,
                bytes: bytes as u64,
                shape,
                dtype,
            });
        }
    }
    StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(slices)
        .unwrap()
}
fn plans(cases: &[Case]) -> Vec<Bf16GemmExPlan> {
    let mut result = Vec::new();
    for c in cases {
        let p = if c.role == TensorRole::RoutedExpertGate {
            MIXED
                .gemmex_gate_up_plan(layout(c), layout(c), c.m, usize::MAX, usize::MAX)
                .unwrap()
        } else {
            MIXED
                .gemmex_plan(&c.role, layout(c), c.m, usize::MAX, usize::MAX)
                .unwrap()
        };
        if let Some(p) = p
            && !result.contains(&p)
        {
            result.push(p);
        }
    }
    result
}
fn owner(
    ops: &Rc<CudaOperators>,
    bound: &BoundStateDict,
    plans: &[Bf16GemmExPlan],
) -> CudaStandardDecoderOperators {
    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        Rc::clone(ops),
        bound.parameters(),
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 32 << 20,
        }),
        8 << 20,
        MIXED,
    )
    .unwrap();
    owner.prepare_bf16_gemmex(plans).unwrap();
    owner
}
fn input(owner: &CudaStandardDecoderOperators, dir: &Path, c: &Case) -> Rows {
    rows_bf16(
        RowsShape::new(c.m, c.k).unwrap(),
        None,
        owner
            .ops
            .upload_bf16_words(&words(&load(dir, c, "input.bf16")))
            .unwrap(),
    )
    .unwrap()
}
fn compare(owner: &CudaStandardDecoderOperators, dir: &Path, c: &Case, y: &Rows) -> (usize, usize) {
    let actual = owner
        .ops
        .download_bf16_buffer(owner.bf16_buffer(y).unwrap())
        .unwrap()
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect::<Vec<_>>();
    let expected = words(&load(dir, c, "hf.bf16"));
    assert_eq!(actual.len(), c.m * c.n);
    assert_eq!(actual.len(), expected.len());
    let unequal = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    eprintln!(
        "GEMMEX_SELECTED case={} m={} n={} k={} unequal={} words={} first={:?}",
        c.name,
        c.m,
        c.n,
        c.k,
        unequal,
        actual.len(),
        actual.iter().zip(&expected).position(|(a, b)| a != b)
    );
    (unequal, actual.len())
}
fn matrix(
    dir: &Path,
    expected_cases: usize,
    expected_words: usize,
    mut report: Option<&mut std::fs::File>,
) {
    let cases = cases(dir);
    assert_eq!(cases.len(), expected_cases);
    let bound = bound(dir, &cases);
    let plans = plans(&cases);
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&ops, &bound, &plans);
    let materializer = StateDictMaterializer::new(64 << 20).unwrap();
    let mut unequal = 0;
    let mut count = 0;
    let mut calls = 0;
    let linear = |i: usize| {
        let p = bound.get_by_id(ParameterId::new(i as u64 + 1)).unwrap();
        materializer.prepared_linear(p, p.role().clone()).unwrap()
    };
    for (i, c) in cases.iter().enumerate() {
        if c.role == TensorRole::RoutedExpertUp || c.role == TensorRole::RoutedExpertDown {
            continue;
        }
        let x = input(&owner, dir, c);
        let results = owner
            .synchronous(|this| {
                if let Some(id) = c.expert {
                    let up = cases
                        .iter()
                        .position(|x| x.expert == Some(id) && x.role == TensorRole::RoutedExpertUp)
                        .unwrap();
                    let down = cases
                        .iter()
                        .position(|x| {
                            x.expert == Some(id) && x.role == TensorRole::RoutedExpertDown
                        })
                        .unwrap();
                    assert_eq!(
                        load(dir, c, "input.bf16"),
                        load(dir, &cases[up], "input.bf16")
                    );
                    let metadata = ExpertMetadata::new(
                        0,
                        id,
                        [i, up, down].map(|j| {
                            bound
                                .get_by_id(ParameterId::new(j as u64 + 1))
                                .unwrap()
                                .clone()
                        }),
                        None,
                    )
                    .unwrap();
                    let (key, bytes, scratch) = this.metadata_plan(&metadata, 0, id, c.m, c.k)?;
                    assert_eq!(key.numeric_precision, Some(MIXED));
                    this.admitted_expert_operation(key, bytes, scratch, |this| {
                        for j in [i, up, down] {
                            this.bindings.linear(&this.ops, &linear(j))?;
                        }
                        let (a, b) = this
                            .bf16_metadata_gemmex_pair(&metadata, &x, None)?
                            .expect("production pair path selected");
                        let mut results = vec![
                            (i, compare(this, dir, c, &a)),
                            (up, compare(this, dir, &cases[up], &b)),
                        ];
                        let down_input = input(this, dir, &cases[down]);
                        let w = this
                            .bindings
                            .metadata_linear(&metadata.parameters()[2])?
                            .unwrap();
                        let y = this.bf16_resident_linear(&w, &down_input, None)?;
                        results.push((down, compare(this, dir, &cases[down], &y)));
                        Ok(results)
                    })
                } else {
                    let y = this.bf16_linear_rows(&linear(i), &x, None)?;
                    Ok(vec![(i, compare(this, dir, c, &y))])
                }
            })
            .unwrap();
        calls += if c.expert.is_some() { 2 } else { 1 };
        for (index, (bad, words)) in results {
            unequal += bad;
            count += words;
            if let Some(report) = report.as_deref_mut() {
                serde_json::to_writer(&mut *report,&serde_json::json!({"case":cases[index].name,"unequal":bad,"words":words,"native_bf16_output":true,"production_model_path":true})).unwrap();
                writeln!(report).unwrap();
                report.sync_all().unwrap();
            }
        }
        assert!(owner.bf16_hold.is_empty());
        assert_eq!(
            owner.bf16_gemmex.as_ref().unwrap().provider_version(),
            120405
        );
    }
    assert_eq!(count, expected_words);
    eprintln!(
        "GEMMEX_POOL storage={} external_reserve={} observed_growth={} version={}",
        owner.bf16_gemmex.as_ref().unwrap().plan().storage_bytes(),
        owner.bf16_gemmex.as_ref().unwrap().plan().provider_bytes(),
        owner
            .bf16_gemmex
            .as_ref()
            .unwrap()
            .provider_observed_growth_bytes(),
        owner.bf16_gemmex.as_ref().unwrap().provider_version()
    );
    owner.quiesce().unwrap();
    drop(owner);
    ops.trim_device_allocator().unwrap();
    assert_eq!(ops.allocator_metrics().live_requested_bytes, baseline);
    eprintln!(
        "GEMMEX_MATRIX cases={} native_calls={calls} unequal={unequal} words={count} scratch_reclaimed=true",
        cases.len()
    );
    assert_eq!(unequal, 0);
}
#[test]
#[ignore = "sole GPU owner; original independently pinned 114 frozen projections; CUDA12.4/sm86"]
fn mixed_gemmex_frozen_114_production_paths() {
    let dir = PathBuf::from(std::env::var_os("FERRULE_GEMMEX_FIXTURE").unwrap());
    assert_eq!(
        std::fs::read_to_string(dir.join("cases.txt"))
            .unwrap()
            .lines()
            .count(),
        114
    );
    let mut report = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(std::env::var_os("FERRULE_GEMMEX_REPORT").unwrap())
        .unwrap();
    matrix(&dir, 114, 209920, Some(&mut report));
}
struct Synthetic(PathBuf);
impl Synthetic {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("mixed-gemmex-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let mut lines = String::new();
        for (name, m) in [
            ("SharedExpertGate", 5),
            ("expert0_gate", 2),
            ("expert0_up", 2),
            ("expert0_down", 2),
        ] {
            lines.push_str(&format!("{name} {m} 256 256\n"));
            for (suffix, data) in [
                ("weight.fp8", vec![0x38; 256 * 256]),
                ("scale.bf16", 0x3f80u16.to_le_bytes().repeat(4)),
                ("input.bf16", 0x3f80u16.to_le_bytes().repeat(m * 256)),
                ("hf.bf16", 0x4380u16.to_le_bytes().repeat(m * 256)),
            ] {
                std::fs::write(dir.join(format!("{name}.{suffix}")), data).unwrap();
            }
        }
        // The synthetic expectation is analytic 256*1, not an HF fixture.
        std::fs::write(dir.join("cases.txt"), lines).unwrap();
        Self(dir)
    }
}
impl Drop for Synthetic {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
#[ignore = "sole GPU owner; >128 model-selected dense/concatenated/routed paths, analytic not HF"]
fn mixed_gemmex_synthetic_256_model_paths() {
    let f = Synthetic::new();
    matrix(&f.0, 4, 2816, None);
}
#[test]
#[ignore = "sole GPU owner; allocator failure before preparation, clean retry and reclaim"]
fn mixed_gemmex_prepare_failure_clean_retry() {
    let f = Synthetic::new();
    let cases = cases(&f.0);
    let bound = bound(&f.0, &cases);
    let plans = plans(&cases);
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&ops, &bound, &[]);
    ops.failpoints().arm_allocation();
    assert!(owner.prepare_bf16_gemmex(&plans).is_err());
    assert!(!owner.needs_quarantine());
    assert!(owner.bf16_gemmex.is_none());
    owner.prepare_bf16_gemmex(&plans).unwrap();
    drop(owner);
    ops.trim_device_allocator().unwrap();
    assert_eq!(ops.allocator_metrics().live_requested_bytes, baseline);
}
fn failure(unknown: bool) {
    let f = Synthetic::new();
    let cases = cases(&f.0);
    let bound = bound(&f.0, &cases);
    let plans = plans(&cases);
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&ops, &bound, &plans);
    let x = input(&owner, &f.0, &cases[0]);
    let p = bound.get_by_id(ParameterId::new(1)).unwrap();
    let linear = StateDictMaterializer::new(1 << 20)
        .unwrap()
        .prepared_linear(p, p.role().clone())
        .unwrap();
    let result = owner.synchronous::<()>(|this| {
        let y = this.bf16_linear_rows(&linear, &x, None)?;
        assert_eq!(compare(this, &f.0, &cases[0], &y).0, 0);
        this.lose_next_consumer_proof = unknown;
        Err(cuda_error("controlled post-submission failure"))
    });
    assert!(result.is_err());
    assert_eq!(owner.needs_quarantine(), unknown);
    eprintln!(
        "GEMMEX_FAILURE unknown={unknown} pool_retained={}",
        owner.bf16_gemmex.is_some()
    );
    if unknown {
        assert!(!owner.bf16_hold.is_empty());
        assert!(!owner.expert_artifacts.is_empty());
        assert!(
            owner
                .synchronous(|this| this.bf16_linear_rows(&linear, &x, None))
                .is_err()
        );
        let live = ops.allocator_metrics().live_requested_bytes;
        drop(x);
        drop(owner);
        ops.sync_stream().unwrap();
        ops.trim_device_allocator().unwrap();
        assert!(ops.allocator_metrics().live_requested_bytes >= live);
        assert!(Rc::strong_count(&ops) > 1);
        eprintln!(
            "GEMMEX_UNKNOWN before_drop={live} after_drop={} reuse_blocked=true operands_retained=true",
            ops.allocator_metrics().live_requested_bytes
        );
    } else {
        assert!(owner.bf16_hold.is_empty());
        let output = owner
            .synchronous(|this| this.bf16_linear_rows(&linear, &x, None))
            .unwrap();
        assert_eq!(compare(&owner, &f.0, &cases[0], &output).0, 0);
        drop(output);
        drop(x);
        drop(owner);
        ops.trim_device_allocator().unwrap();
        assert_eq!(ops.allocator_metrics().live_requested_bytes, baseline);
    }
}
#[test]
#[ignore = "sole GPU owner; actual native submission then clean error/reuse/reclaim"]
fn mixed_gemmex_clean_failure_reuses_pool() {
    failure(false)
}
#[test]
#[ignore = "sole GPU owner; actual native submission then Unknown custody; fresh process"]
fn mixed_gemmex_unknown_retains_pool_and_operands() {
    failure(true)
}

#[test]
fn mixed_gemmex_synthetic_binding_metadata_cpu() {
    let f = Synthetic::new();
    let cases = cases(&f.0);
    let bound = bound(&f.0, &cases);
    assert_eq!(bound.parameters().len(), 4);
    let plans = plans(&cases);
    assert_eq!(plans.len(), 3);
    let materializer = StateDictMaterializer::new(1 << 20).unwrap();
    for p in bound.parameters() {
        let linear = materializer.prepared_linear(p, p.role().clone()).unwrap();
        assert!(linear.numeric_fp8().is_some());
        assert_eq!(linear.global_shape(), [256, 256]);
    }
    let metadata = ExpertMetadata::new(
        0,
        0,
        [2, 3, 4].map(|id| bound.get_by_id(ParameterId::new(id)).unwrap().clone()),
        None,
    )
    .unwrap();
    assert_eq!(metadata.expected_device_bytes(), 3 * (256 * 256 + 8));
    assert_eq!(metadata.input_width(), 256);
}
#[test]
#[ignore = "CPU frozen source gate; set FERRULE_GEMMEX_FIXTURE, hide GPUs"]
fn mixed_gemmex_frozen_114_metadata_cpu() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let dir = PathBuf::from(std::env::var_os("FERRULE_GEMMEX_FIXTURE").unwrap());
    let cases = cases(&dir);
    let bound = bound(&dir, &cases);
    assert_eq!(cases.len(), 114);
    assert_eq!(bound.parameters().len(), 114);
    assert_eq!(cases.iter().map(|c| c.m * c.n).sum::<usize>(), 209920);
    let mut native_calls = 0;
    for c in &cases {
        assert_eq!(load(&dir, c, "weight.fp8").len(), c.n * c.k);
        assert_eq!(load(&dir, c, "scale.bf16").len(), c.n * c.k / 8192);
        assert_eq!(load(&dir, c, "hf.bf16").len(), c.m * c.n * 2);
        assert_eq!(load(&dir, c, "decoded.bf16").len(), c.n * c.k * 2);
        if c.role == TensorRole::RoutedExpertGate {
            let up = cases
                .iter()
                .find(|p| p.expert == c.expert && p.role == TensorRole::RoutedExpertUp)
                .unwrap();
            assert_eq!(c.m, up.m);
            assert_eq!(layout(c), layout(up));
            assert_eq!(load(&dir, c, "input.bf16"), load(&dir, up, "input.bf16"));
        }
        if c.role != TensorRole::RoutedExpertUp {
            native_calls += 1;
        }
    }
    assert_eq!(native_calls, 78);
    assert!(
        plans(&cases)
            .iter()
            .any(|p| p.rows() == 2 && p.layout().n == 1024)
    );
}

#[path = "resident.rs"]
mod resident;

mod resident_control;

mod resident_scalar;
