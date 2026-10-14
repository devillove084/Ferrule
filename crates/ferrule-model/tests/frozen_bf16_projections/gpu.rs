use super::*;
use ferrule_backend::cuda::operators::linear::{CudaOperators, F32ToBf16RowsLayout};
use std::io::Write;

fn words(data: &[u8]) -> Vec<u16> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|w| u16::from_le_bytes([w[0], w[1]]))
        .collect()
}

fn emit(report: &mut std::fs::File, value: serde_json::Value) {
    serde_json::to_writer(&mut *report, &value).unwrap();
    writeln!(report).unwrap();
    report.sync_all().unwrap();
}

#[test]
#[ignore = "GPU owner only; 12 frozen independent HF pairs x five local plans, no full40"]
fn frozen_projection_operator_matrix() {
    let (dir, manifest) = fixture();
    assert_eq!(manifest.schema, "ferrule.frozen-bf16-projections.v1");
    run_matrix(dir, manifest);
}

pub(super) fn run_matrix(dir: PathBuf, manifest: Manifest) {
    let pairs = manifest.cases.len();
    let missing = manifest.missing_expert_pairs;
    let additional_selected_rows = manifest
        .cases
        .iter()
        .filter(|c| {
            MODEL
                .linear_plan(&c.role, layout(c.n, c.k), 1, MODEL_BUDGET)
                .unwrap()
                .selected_kernel()
                == K::Bf16M1Gemv
        })
        .count();
    let pin = if manifest.schema == "ferrule.frozen-bf16-layers234.v1" {
        gqa_boundary::PIN
    } else {
        FIXTURE_PIN
    };
    let report_path =
        std::env::var_os("FERRULE_FROZEN_BF16_REPORT").expect("fresh JSONL report required");
    let mut report = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&report_path)
        .unwrap();
    emit(
        &mut report,
        serde_json::json!({
            "event": "start", "fixture_sha256": pin, "pairs": pairs, "diagnostic_alternatives": pairs * 5,
            "additional_selected_rows": additional_selected_rows,
            "missing_expert_pairs": missing, "model_profile": MODEL.hybrid_backend_name(),
            "input_source": "original pinned HF BF16 hook bytes, never production replay",
            "atol": 0.00025, "rtol": 0.012,
            "tolerance_use": "unchanged tiny bound diagnostic only; compatibility requires exact BF16",
            "model_acceptance": false
        }),
    );
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut profile_exact_cases = 0;
    let mut any_candidate_exact_cases = 0;
    let mut comparisons = 0;
    let mut resources_ok = true;
    let mut all_finite = true;
    for c in &manifest.cases {
        let l = layout(c.n, c.k);
        let artifact = op
            .upload_numeric_fp8_linear(
                l,
                &bytes(&dir, c, "weight.fp8", c.n * c.k),
                &bytes(
                    &dir,
                    c,
                    "scale.bf16",
                    c.n.div_ceil(128) * c.k.div_ceil(128) * 2,
                ),
            )
            .unwrap();
        let input = op
            .upload_bf16_words(&words(&bytes(&dir, c, "input.bf16", c.k * 2)))
            .unwrap();
        let expected = words(&bytes(&dir, c, "hf.bf16", c.n * 2));
        let mut output = op.zero_f32_buffer(c.n).unwrap();
        let mut rounded = op.zero_bf16_buffer(c.n).unwrap();
        let selected = MODEL.linear_plan(&c.role, l, 1, MODEL_BUDGET).unwrap();
        let mut any_exact = false;
        let mut alternatives = matrix(c.k).to_vec();
        if selected.selected_kernel() == K::Bf16M1Gemv {
            alternatives.push((selected.precision(), MODEL_BUDGET));
        }
        for (precision, budget) in alternatives {
            let plan = NumericFp8LinearPlan::new(l, 1, budget, precision).unwrap();
            // One bounded workspace at a time; fence before each replacement/drop.
            op.prepare_numeric_fp8_provider(plan, plan.provider_bytes())
                .unwrap();
            let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
            emit(
                &mut report,
                serde_json::json!({"event": "submission", "case": c.id,
                "precision": format!("{precision:?}"), "budget": budget}),
            );
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
                F32ToBf16RowsLayout::packed(1, 1, c.n).unwrap(),
            )
            .unwrap();
            let counts = op.counters();
            let after = op.allocator_metrics();
            let clean = counts.compute_kernel_launches == plan.kernel_launches() as u64 + 1
                && counts.device_allocation_attempts == 0
                && counts.stream_wide_syncs == 0
                && before.live_granted_bytes == after.live_granted_bytes
                && before.driver_allocations == after.driver_allocations
                && !workspace.is_poisoned();
            resources_ok &= clean;
            let raw = op.download_f32_buffer(&output).unwrap();
            let actual = op.download_bf16_buffer(&rounded).unwrap();
            let mut unequal = 0;
            let mut outside = 0;
            let mut nonfinite = 0;
            let mut max_abs = 0.0_f64;
            let mut first = None;
            for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                let e = f32::from_bits(u32::from(e) << 16);
                let finite = a.is_finite() && raw[i].is_finite();
                nonfinite += usize::from(!finite);
                let error = (f64::from(a) - f64::from(e)).abs();
                if finite {
                    max_abs = max_abs.max(error);
                }
                outside += usize::from(!finite || error > 0.00025 + 0.012 * f64::from(e).abs());
                if a.to_bits() != e.to_bits() {
                    unequal += 1;
                    if first.is_none() {
                        first = Some(
                            serde_json::json!({"index": i, "actual": a, "hf": e, "f32_accumulator": raw[i]}),
                        );
                    }
                }
            }
            all_finite &= nonfinite == 0;
            let exact = unequal == 0 && nonfinite == 0;
            let is_profile = (budget == MODEL_BUDGET || budget == 0)
                && plan.selected_kernel() == selected.selected_kernel();
            if is_profile && exact {
                profile_exact_cases += 1;
            }
            if budget != 0 {
                any_exact |= exact;
            }
            comparisons += 1;
            emit(
                &mut report,
                serde_json::json!({
                    "event": "comparison", "case": c.id, "role": format!("{:?}", c.role),
                    "shape": [1, c.n, c.k], "precision": format!("{precision:?}"),
                    "kernel": plan.selected_kernel().name(), "budget": budget, "tile_rows": plan.tile_rows(),
                    "current_profile_selection": is_profile, "production_trace_sameinput": c.production_sameinput,
                    "elements": c.n, "unequal_hf": unequal, "outside_existing_bound": outside,
                    "nonfinite": nonfinite, "max_abs": max_abs, "first_difference": first,
                    "workspace_bytes": plan.workspace_requirements().bytes,
                    "provider_reserve_bytes": plan.provider_bytes(),
                    "provider_observed_growth_bytes": workspace.provider_observed_growth_bytes(),
                    "provider_version": workspace.provider_version(),
                    "workspace_grant_bytes": plan.workspace_allocation_grant_bytes().unwrap(),
                    "artifact_grant_bytes": plan.artifact_allocation_grant_bytes().unwrap(),
                    "live_granted_bytes": after.live_granted_bytes, "resources_ok": clean,
                    "compute_launches": counts.compute_kernel_launches,
                    "device_allocation_attempts": counts.device_allocation_attempts,
                    "stream_wide_syncs": counts.stream_wide_syncs
                }),
            );
            op.record_compute_event().unwrap().synchronize().unwrap();
        }
        any_candidate_exact_cases += usize::from(any_exact);
    }
    emit(
        &mut report,
        serde_json::json!({"event": "complete", "comparisons": comparisons,
        "profile_exact_cases": profile_exact_cases, "any_candidate_exact_cases": any_candidate_exact_cases,
        "independent_cases": pairs, "missing_expert_pairs": missing, "model_acceptance": false,
        "resources_ok": resources_ok, "all_finite": all_finite}),
    );
    assert_eq!(comparisons, pairs * 5 + additional_selected_rows);
    assert!(
        resources_ok && all_finite,
        "resource/nonfinite failure: {report_path:?}"
    );
    assert_eq!(
        profile_exact_cases, pairs,
        "local mixed model schedule is not bitwise compatible for all frozen pairs; this alone is not a numerical-defect claim; report: {report_path:?}"
    );
}

#[test]
#[ignore = "GPU owner only; selected Mixed M5 QKV, pinned independent HF fixture, no full40"]
fn mixed_m5_qkv_selected_schedule() {
    let dir = PathBuf::from(std::env::var_os("FERRULE_BF16_M5_DIR").unwrap());
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
        "a586606e89ea99537b601a2abefd9d0636be3a8085e4f0bc30ee3c4b640a31c7"
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let read = |name: &str, len: usize| {
        let data = std::fs::read(dir.join(name)).unwrap();
        assert_eq!(data.len(), len);
        data
    };
    let (m, n, k) = (5, 8192, 2048);
    let op = CudaOperators::new_on_device(0).unwrap();
    let artifact = op
        .upload_numeric_fp8_linear(
            layout(n, k),
            &read("weight.fp8", n * k),
            &read("scale.bf16", 2048),
        )
        .unwrap();
    let input = op
        .upload_bf16_words(&words(&read("input.bf16", m * k * 2)))
        .unwrap();
    let expected = words(&read("hf.bf16", m * n * 2));
    let mut output = op.zero_f32_buffer(m * n).unwrap();
    let mut rounded = op.zero_bf16_buffer(m * n).unwrap();
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(std::env::var_os("FERRULE_FROZEN_BF16_REPORT").unwrap())
        .unwrap();
    let mut exact = 0;
    let mut resources_ok = true;
    for tile in [64, 257, 1024, 2043] {
        let budget = (m + tile) * k * 2;
        let plan = MODEL
            .linear_plan(&TensorRole::LinearAttentionQkv, layout(n, k), m, budget)
            .unwrap();
        assert_eq!(plan.selected_kernel(), K::Bf16TiledSlicedK2Gemm);
        assert_eq!(plan.tile_rows(), tile);
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        emit(
            &mut report,
            serde_json::json!({"event": "submission", "shape": [m,n,k],
            "kernel": plan.selected_kernel().name(), "tile_rows": tile, "budget": budget}),
        );
        op.reset_counters();
        let before = op.allocator_metrics();
        op.numeric_fp8_bf16_linear_into(&artifact, &input, &mut output, &mut workspace, plan, k, n)
            .unwrap();
        op.f32_rows_to_bf16_rne_into(
            &output,
            &mut rounded,
            F32ToBf16RowsLayout::packed(1, m, n).unwrap(),
        )
        .unwrap();
        let counts = op.counters();
        let after = op.allocator_metrics();
        let clean = counts.compute_kernel_launches == plan.kernel_launches() as u64 + 1
            && counts.device_allocation_attempts == 0
            && counts.stream_wide_syncs == 0
            && before.live_granted_bytes == after.live_granted_bytes
            && before.driver_allocations == after.driver_allocations
            && !workspace.is_poisoned();
        resources_ok &= clean;
        let actual = op.download_bf16_buffer(&rounded).unwrap();
        assert_eq!(actual.len(), expected.len());
        let unequal = actual
            .iter()
            .zip(&expected)
            .filter(|(a, e)| a.to_bits() != u32::from(**e) << 16 || !a.is_finite())
            .count();
        exact += usize::from(unequal == 0);
        emit(
            &mut report,
            serde_json::json!({"event": "comparison", "shape": [m,n,k],
            "kernel": plan.selected_kernel().name(), "tile_rows": tile, "elements": m*n,
            "unequal_hf": unequal, "current_profile_selection": true, "resources_ok": clean,
            "workspace_bytes": plan.workspace_requirements().bytes,
            "workspace_grant_bytes": plan.workspace_allocation_grant_bytes().unwrap(),
            "compute_launches": counts.compute_kernel_launches}),
        );
        op.record_compute_event().unwrap().synchronize().unwrap();
    }
    emit(
        &mut report,
        serde_json::json!({"event": "complete", "comparisons": 4,
        "profile_exact_cases": exact, "resources_ok": resources_ok, "model_acceptance": false}),
    );
    assert!(resources_ok);
    assert_eq!(exact, 4);
}
