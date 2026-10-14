use super::*;

const PIN: &str = "0381178a03f7c848a73b153ddf3cec3284c316814ef3007f3f7a8043f96e984f";
#[derive(Deserialize)]
struct CapitalCase {
    id: String,
    role: TensorRole,
    m: usize,
    n: usize,
    k: usize,
}
#[derive(Deserialize)]
struct CapitalManifest {
    schema: String,
    hf_sha256: String,
    original_boundaries_exact: usize,
    supplemental_experts_aggregate_exact: bool,
    cases: Vec<CapitalCase>,
}
fn load() -> (PathBuf, CapitalManifest) {
    let dir = PathBuf::from(std::env::var_os("FERRULE_CAPITAL_PROJECTION_DIR").unwrap());
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
        PIN
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "--status", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let m: CapitalManifest =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m.schema, "ferrule.capital-projection-matrix.v1");
    assert_eq!(
        m.hf_sha256,
        "6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07"
    );
    assert_eq!(m.original_boundaries_exact, 30);
    assert!(m.supplemental_experts_aggregate_exact);
    assert_eq!(m.cases.len(), 114);
    (dir, m)
}
fn choices(m: usize, k: usize) -> Vec<(P, usize)> {
    vec![
        (P::W8A16Bf16F32Accumulate, 0),
        (P::Bf16RneF32Accumulate, (m + 257) * k * 2),
        (P::Bf16SlicedK2F32Accumulate, (m + 257) * k * 2),
        (P::Bf16RneF32Accumulate, MODEL_BUDGET),
        (P::Bf16SlicedK2F32Accumulate, MODEL_BUDGET),
    ]
}
#[test]
fn capital_shapes_have_explicit_admission_and_no_multirow_gemv() {
    for m in [1, 2, 5] {
        for (n, k) in [
            (8192, 2048),
            (4096, 2048),
            (2048, 4096),
            (512, 2048),
            (2048, 512),
        ] {
            for (p, b) in choices(m, k) {
                let plan = NumericFp8LinearPlan::new(layout(n, k), m, b, p).unwrap();
                assert_eq!(plan.rows(), m);
            }
            if m != 1 || (n, k) != (2048, 4096) {
                assert!(
                    NumericFp8LinearPlan::new(
                        layout(n, k),
                        m,
                        MODEL_BUDGET,
                        P::Bf16M1GemvF32Accumulate
                    )
                    .is_err()
                );
            }
        }
    }
}
#[test]
fn capital_frozen_coverage() {
    if std::env::var_os("FERRULE_CAPITAL_PROJECTION_DIR").is_none() {
        return;
    }
    let (_, m) = load();
    let mut seen = BTreeSet::new();
    for c in m.cases {
        assert!(seen.insert(c.id));
        assert!([1, 2, 5].contains(&c.m));
        let expected = match c.role {
            TensorRole::RoutedExpertGate | TensorRole::RoutedExpertUp => Some((512, 2048)),
            TensorRole::RoutedExpertDown => Some((2048, 512)),
            _ => shape(&c.role),
        };
        assert_eq!(expected, Some((c.n, c.k)));
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "sole GPU owner; all Capital L0 numeric projections, no full40"]
fn frozen_capital_projection_matrix() {
    use ferrule_backend::cuda::operators::linear::{CudaOperators, F32ToBf16RowsLayout};
    use std::io::Write;
    let (dir, manifest) = load();
    let mut report = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(std::env::var_os("FERRULE_CAPITAL_PROJECTION_REPORT").unwrap())
        .unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut selected_exact = 0;
    let mut candidate_exact = 0;
    for c in &manifest.cases {
        let read = |suffix: &str, len: usize| {
            let d = std::fs::read(dir.join(format!("{}.{}", c.id, suffix))).unwrap();
            assert_eq!(d.len(), len);
            d
        };
        let words = |data: Vec<u8>| {
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|w| u16::from_le_bytes(*w))
                .collect::<Vec<_>>()
        };
        let l = layout(c.n, c.k);
        let artifact = op
            .upload_numeric_fp8_linear(
                l,
                &read("weight.fp8", c.n * c.k),
                &read("scale.bf16", c.n.div_ceil(128) * c.k.div_ceil(128) * 2),
            )
            .unwrap();
        let x = words(read("input.bf16", c.m * c.k * 2));
        let hf = words(read("hf.bf16", c.m * c.n * 2));
        let input = op.upload_bf16_words(&x).unwrap();
        let mut output = op.zero_f32_buffer(c.m * c.n).unwrap();
        let mut rounded = op.zero_bf16_buffer(c.m * c.n).unwrap();
        let native = read("native.f32", c.m * c.n * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        let selected = MODEL.linear_plan(&c.role, l, c.m, MODEL_BUDGET).unwrap();
        let mut any = false;
        let mut chosen = false;
        for (precision, budget) in choices(c.m, c.k) {
            let plan = NumericFp8LinearPlan::new(l, c.m, budget, precision).unwrap();
            let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
            op.reset_counters();
            let before = op.allocator_metrics();
            op.numeric_fp8_bf16_linear_into(
                &artifact,
                &input,
                &mut output,
                &mut workspace,
                plan,
                c.k,
                c.n,
            )
            .unwrap();
            op.f32_rows_to_bf16_rne_into(
                &output,
                &mut rounded,
                F32ToBf16RowsLayout::packed(1, c.m, c.n).unwrap(),
            )
            .unwrap();
            let count = op.counters();
            let after = op.allocator_metrics();
            assert_eq!(
                count.compute_kernel_launches,
                plan.kernel_launches() as u64 + 1
            );
            assert_eq!(count.device_allocation_attempts, 0);
            assert_eq!(count.stream_wide_syncs, 0);
            assert_eq!(before.live_granted_bytes, after.live_granted_bytes);
            assert_eq!(before.driver_allocations, after.driver_allocations);
            assert!(!workspace.is_poisoned());
            let raw = op.download_f32_buffer(&output).unwrap();
            let actual = op.download_bf16_buffer(&rounded).unwrap();
            let mut diff = Vec::new();
            let mut raw_diff = 0;
            let mut max_abs = 0f32;
            for i in 0..hf.len() {
                assert!(raw[i].is_finite() && actual[i].is_finite());
                let a = (actual[i].to_bits() >> 16) as u16;
                let b = raw[i].to_bits();
                assert_eq!(a, (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16);
                raw_diff += usize::from(b != native[i]);
                if a != hf[i] {
                    diff.push(serde_json::json!({"index":i,"actual":format!("{a:04x}"),"hf":format!("{:04x}",hf[i]),"raw":format!("{b:08x}"),"native_raw":format!("{:08x}",native[i])}));
                    max_abs =
                        max_abs.max((actual[i] - f32::from_bits(u32::from(hf[i]) << 16)).abs());
                }
            }
            let is_selected =
                plan.precision() == selected.precision() && (budget == 0 || budget == MODEL_BUDGET);
            any |= diff.is_empty();
            chosen |= is_selected && diff.is_empty();
            serde_json::to_writer(&mut report,&serde_json::json!({"event":"comparison","case":c.id,"role":format!("{:?}",c.role),"shape":[c.m,c.n,c.k],"precision":format!("{precision:?}"),"tile":plan.tile_rows(),"budget":budget,"selected":is_selected,"bf16_unequal":diff.len(),"raw_unequal":raw_diff,"max_abs":max_abs,"differences":diff,"resources_ok":true})).unwrap();
            writeln!(report).unwrap();
            report.sync_all().unwrap();
            op.record_compute_event().unwrap().synchronize().unwrap();
        }
        candidate_exact += usize::from(any);
        selected_exact += usize::from(chosen);
    }
    writeln!(report,"{}",serde_json::json!({"event":"complete","cases":manifest.cases.len(),"any_exact":candidate_exact,"selected_exact":selected_exact})).unwrap();
    report.sync_all().unwrap();
    if std::env::var_os("FERRULE_CAPITAL_REQUIRE_SELECTED_EXACT").is_some() {
        assert_eq!(selected_exact, manifest.cases.len());
    }
}
