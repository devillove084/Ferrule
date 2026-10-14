//! Frozen independent HF inputs/outputs; operator compatibility, not model acceptance.
use ferrule_backend::cuda_plan::{
    NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision as P, NumericFp8ScaleType,
    NumericFp8SelectedKernel as K,
};
use ferrule_model::support::TensorRole;
use ferrule_model::transformer::NumericFp8Precision as ModelPrecision;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const FIXTURE_PIN: &str = "59684d1793711f31dd168c5a9154f93a1cddebfeb292ba11c6dcdead62b5bff8";
const HF_PIN: &str = "41745fc2edaa2616e0018c7311144ea19bd6106fd5bb66835a7c0b800664d071";
const PROFILE_PIN: &str = "351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9";
const MODEL_BUDGET: usize = 8 << 20;
const MODEL: ModelPrecision = ModelPrecision::Bf16ModelMixedF32Accumulate;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    oracle: String,
    hf_sha256: String,
    profile_sha256: String,
    torch: String,
    transformers: String,
    model_boundary: String,
    accumulation: String,
    allow_tf32: bool,
    allow_bf16_reduced_precision_reduction: bool,
    numeric_count: usize,
    missing_expert_pairs: usize,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    layer: usize,
    role: TensorRole,
    n: usize,
    k: usize,
    production_sameinput: bool,
}

fn shape(role: &TensorRole) -> Option<(usize, usize)> {
    match role {
        TensorRole::LinearAttentionQkv => Some((8192, 2048)),
        TensorRole::LinearAttentionZ => Some((4096, 2048)),
        TensorRole::AttentionOutput => Some((2048, 4096)),
        TensorRole::SharedExpertGate | TensorRole::SharedExpertUp => Some((512, 2048)),
        TensorRole::SharedExpertDown => Some((2048, 512)),
        _ => None,
    }
}

fn roles() -> [TensorRole; 6] {
    [
        TensorRole::LinearAttentionQkv,
        TensorRole::LinearAttentionZ,
        TensorRole::AttentionOutput,
        TensorRole::SharedExpertGate,
        TensorRole::SharedExpertUp,
        TensorRole::SharedExpertDown,
    ]
}

fn layout(n: usize, k: usize) -> NumericFp8Layout {
    NumericFp8Layout {
        n,
        k,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    }
}

fn matrix(k: usize) -> [(P, usize); 5] {
    [
        (P::W8A16Bf16F32Accumulate, 0),
        (P::Bf16RneF32Accumulate, (1 + 257) * k * 2),
        (P::Bf16SlicedK2F32Accumulate, (1 + 257) * k * 2),
        (P::Bf16RneF32Accumulate, MODEL_BUDGET),
        (P::Bf16SlicedK2F32Accumulate, MODEL_BUDGET),
    ]
}

impl Manifest {
    fn validate(&self) -> Result<(), &'static str> {
        if self.schema == "ferrule.frozen-bf16-layers234.v1" {
            return gqa_boundary::validate(self);
        }
        if self.schema != "ferrule.frozen-bf16-projections.v1"
            || self.oracle != "independent-pinned-hf-bf16"
            || self.hf_sha256 != HF_PIN
            || self.profile_sha256 != PROFILE_PIN
            || self.torch != "2.13.0+cu129"
            || self.transformers != "5.5.3"
            || self.model_boundary != "bf16"
            || self.accumulation != "f32"
            || self.allow_tf32
            || self.allow_bf16_reduced_precision_reduction
        {
            return Err("unknown oracle or precision metadata");
        }
        if self.numeric_count != 60 || self.missing_expert_pairs != 48 || self.cases.len() != 12 {
            return Err("incomplete or invented independent coverage");
        }
        let mut identities = BTreeSet::new();
        for c in &self.cases {
            if c.layer > 1
                || shape(&c.role) != Some((c.n, c.k))
                || c.id != format!("l{}_{:?}", c.layer, c.role)
                || !identities.insert(c.id.clone())
            {
                return Err("unknown, duplicate or malformed projection");
            }
            let same = c.layer == 0
                && matches!(
                    c.role,
                    TensorRole::LinearAttentionQkv
                        | TensorRole::LinearAttentionZ
                        | TensorRole::AttentionOutput
                );
            if c.production_sameinput != same {
                return Err("incorrect SAMEINPUT diagnostic");
            }
        }
        Ok(())
    }
}

fn fixture() -> (PathBuf, Manifest) {
    let dir = PathBuf::from(
        std::env::var_os("FERRULE_FROZEN_BF16_DIR").expect("FERRULE_FROZEN_BF16_DIR required"),
    );
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    manifest.validate().unwrap();
    let expected_pin = if manifest.schema == "ferrule.frozen-bf16-layers234.v1" {
        gqa_boundary::PIN
    } else {
        FIXTURE_PIN
    };
    let hash = Command::new("sha256sum")
        .arg(dir.join("SHA256SUMS"))
        .output()
        .unwrap();
    assert!(hash.status.success());
    assert_eq!(
        String::from_utf8(hash.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        expected_pin
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    (dir, manifest)
}

fn bytes(dir: &Path, c: &Case, suffix: &str, len: usize) -> Vec<u8> {
    let data = std::fs::read(dir.join(format!("{}.{}", c.id, suffix))).unwrap();
    assert_eq!(data.len(), len);
    data
}

fn sample() -> Manifest {
    Manifest {
        schema: "ferrule.frozen-bf16-projections.v1".into(),
        oracle: "independent-pinned-hf-bf16".into(),
        hf_sha256: HF_PIN.into(),
        profile_sha256: PROFILE_PIN.into(),
        torch: "2.13.0+cu129".into(),
        transformers: "5.5.3".into(),
        model_boundary: "bf16".into(),
        accumulation: "f32".into(),
        allow_tf32: false,
        allow_bf16_reduced_precision_reduction: false,
        numeric_count: 60,
        missing_expert_pairs: 48,
        cases: (0..2)
            .flat_map(|layer| {
                roles().into_iter().enumerate().map(move |(index, role)| {
                    let (n, k) = shape(&role).unwrap();
                    Case {
                        id: format!("l{layer}_{role:?}"),
                        layer,
                        role,
                        n,
                        k,
                        production_sameinput: layer == 0 && index < 3,
                    }
                })
            })
            .collect(),
    }
}

#[test]
fn all_numeric_shape_families_admit_bounded_matrix() {
    for (n, k) in [
        (8192, 2048),
        (4096, 2048),
        (2048, 4096),
        (512, 2048),
        (2048, 512),
    ] {
        let l = layout(n, k);
        for (p, budget) in matrix(k) {
            let plan = NumericFp8LinearPlan::new(l, 1, budget, p).unwrap();
            assert!(plan.workspace_requirements().bytes <= budget as u64);
            if budget != 0 {
                assert_eq!(plan.tile_rows(), n.min(budget / (k * 2) - 1));
                assert_eq!(
                    plan.workspace_requirements().bytes as usize,
                    (1 + plan.tile_rows()) * k * 2
                );
                let other =
                    NumericFp8LinearPlan::new(l, 1, budget, P::Bf16SlicedK2F32Accumulate).unwrap();
                assert_eq!(
                    plan.workspace_requirements(),
                    other.workspace_requirements()
                );
                assert_eq!(
                    plan.workspace_allocation_grant_bytes().unwrap(),
                    other.workspace_allocation_grant_bytes().unwrap()
                );
            }
        }
    }
}

#[test]
fn semantic_schedule_and_missing_experts_are_not_shape_equivalence_claims() {
    for role in roles().into_iter().chain([
        TensorRole::RoutedExpertGate,
        TensorRole::RoutedExpertUp,
        TensorRole::RoutedExpertDown,
    ]) {
        let (n, k) = shape(&role).unwrap_or(if role == TensorRole::RoutedExpertDown {
            (2048, 512)
        } else {
            (512, 2048)
        });
        for m in [1, 2, 5, 9] {
            let p = MODEL
                .linear_plan(&role, layout(n, k), m, MODEL_BUDGET)
                .unwrap();
            let expected = if (role == TensorRole::LinearAttentionZ && m == 1)
                || (role == TensorRole::LinearAttentionQkv && m > 1)
            {
                K::Bf16TiledSlicedK2Gemm
            } else if m == 1 && role == TensorRole::AttentionOutput {
                K::Bf16M1Gemv
            } else if m == 1 && role != TensorRole::LinearAttentionQkv {
                K::W8A16FusedBf16Vector
            } else {
                K::Bf16TiledGemm
            };
            assert_eq!(p.selected_kernel(), expected);
        }
    }
    assert!(
        NumericFp8LinearPlan::new(
            layout(512, 2048),
            2,
            MODEL_BUDGET,
            P::Bf16SlicedK2F32Accumulate
        )
        .is_ok()
    );
    assert!(
        NumericFp8LinearPlan::new(
            layout(512, 2047),
            1,
            MODEL_BUDGET,
            P::Bf16SlicedK2F32Accumulate
        )
        .is_err()
    );
    assert_eq!(
        MODEL
            .linear_plan(
                &TensorRole::LinearAttentionZ,
                layout(512, 2047),
                1,
                MODEL_BUDGET
            )
            .unwrap()
            .selected_kernel(),
        K::Bf16TiledGemm
    );
}

#[test]
fn metadata_rejects_unknown_oracle_and_precision() {
    assert!(sample().validate().is_ok());
    let mut m = sample();
    m.oracle = "production-replay".into();
    assert!(m.validate().is_err());
    let mut m = sample();
    m.allow_bf16_reduced_precision_reduction = true;
    assert!(m.validate().is_err());
    let mut m = sample();
    m.profile_sha256 = "unknown".into();
    assert!(m.validate().is_err());
    assert!(serde_json::from_str::<Case>(r#"{"id":"x","layer":0,"role":"AttentionOutput","n":2048,"k":4096,"production_sameinput":true,"unknown":0}"#).is_err());
}

#[test]
fn metadata_rejects_missing_duplicate_expert_or_wrong_shape_cases() {
    let mut m = sample();
    m.cases.pop();
    assert!(m.validate().is_err());
    let mut m = sample();
    m.cases[1].role = TensorRole::LinearAttentionQkv;
    m.cases[1].id = m.cases[0].id.clone();
    m.cases[1].n = 8192;
    assert!(m.validate().is_err());
    let mut m = sample();
    m.cases[3].role = TensorRole::RoutedExpertGate;
    assert!(m.validate().is_err());
    let mut m = sample();
    m.cases[2].k = 2048;
    assert!(m.validate().is_err());
    let mut m = sample();
    m.cases[6].production_sameinput = true;
    assert!(m.validate().is_err());
}

#[test]
#[ignore = "CPU-only external frozen fixture verification; explicit directory required"]
fn frozen_fixture_integrity() {
    let (dir, m) = fixture();
    for c in &m.cases {
        bytes(&dir, c, "weight.fp8", c.n * c.k);
        bytes(
            &dir,
            c,
            "scale.bf16",
            c.n.div_ceil(128) * c.k.div_ceil(128) * 2,
        );
        for (suffix, len) in [("input.bf16", c.k), ("hf.bf16", c.n)] {
            for w in bytes(&dir, c, suffix, len * 2).as_chunks::<2>().0 {
                assert!(
                    f32::from_bits(u32::from(u16::from_le_bytes([w[0], w[1]])) << 16).is_finite()
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod gpu;

#[path = "../bf16_gqa_boundary/regression.rs"]
mod gqa_boundary;

#[cfg(feature = "cuda")]
#[path = "../bf16_gqa_boundary/kv_projection.rs"]
mod kv_projection;

#[path = "capital.rs"]
mod capital;
