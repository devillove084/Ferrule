//! Fixed supplemental HF layers2..4; reuse the original bounded operator runner.
use super::*;
pub(super) const PIN: &str = "cfa1c5613c179334bcafeacf3dc19dacd9ec947711b29ac701d3a155da1ce63d";
const HF: &str = "5b503e15030f9bd523c293f07c01ad9edde2c7d32b4497745f00c7eaa0b2dbe8";
pub(super) fn validate(m: &Manifest) -> Result<(), &'static str> {
    if m.schema != "ferrule.frozen-bf16-layers234.v1"
        || m.oracle != "independent-pinned-hf-bf16"
        || m.hf_sha256 != HF
        || m.profile_sha256 != PROFILE_PIN
        || m.torch != "2.13.0+cu129"
        || m.transformers != "5.5.3"
        || m.model_boundary != "bf16"
        || m.accumulation != "f32"
        || m.allow_tf32
        || m.allow_bf16_reduced_precision_reduction
        || m.numeric_count != 91
        || m.missing_expert_pairs != 72
        || m.cases.len() != 19
    {
        return Err("unknown supplemental oracle/precision/coverage");
    }
    let mut ids = BTreeSet::new();
    for c in &m.cases {
        let expected = if c.layer == 3 {
            match c.role {
                TensorRole::AttentionQuery => Some((8192, 2048)),
                TensorRole::AttentionKey | TensorRole::AttentionValue => Some((512, 2048)),
                TensorRole::LinearAttentionQkv | TensorRole::LinearAttentionZ => None,
                _ => shape(&c.role),
            }
        } else if c.layer == 2 || c.layer == 4 {
            shape(&c.role)
        } else {
            None
        };
        if expected != Some((c.n, c.k))
            || c.production_sameinput
            || c.id != format!("l{}_{:?}", c.layer, c.role)
            || !ids.insert(c.id.clone())
        {
            return Err("invalid supplemental projection; cannot invent SAMEINPUT trace evidence");
        }
    }
    Ok(())
}

#[test]
fn supplemental_schema_cannot_reinterpret_first_two_layer_oracle() {
    let mut m = sample();
    m.schema = "ferrule.frozen-bf16-layers234.v1".into();
    assert!(m.validate().is_err());
    assert!(validate(&m).is_err());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "sole GPU owner; 19 independently pinned layers2..4 projections, same bounded matrix"]
fn layers234_projection_operator_matrix() {
    let (dir, manifest) = fixture();
    validate(&manifest).unwrap();
    super::gpu::run_matrix(dir, manifest);
}
