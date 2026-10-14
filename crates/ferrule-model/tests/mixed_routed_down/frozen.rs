// Frozen real TP2 expert-down operator gate; no model load or oracle regeneration.
// A pinned external fixture is mandatory. This is not full-program acceptance.
use std::{path::PathBuf, process::Command};

struct Frozen {
    n: usize,
    k: usize,
    ranks: Vec<Rank>,
}
struct Rank {
    input: Vec<u16>,
    weight: Vec<u8>,
    scales: Vec<u8>,
    partial: Vec<u16>,
}
fn words(bytes: &[u8]) -> Vec<u16> {
    assert!(bytes.len().is_multiple_of(2));
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn load() -> Frozen {
    let root = PathBuf::from(
        std::env::var_os("FERRULE_MIXED_DOWN_FIXTURE").expect("frozen TP2 expert-down directory"),
    );
    let pin =
        std::env::var("FERRULE_MIXED_DOWN_SHA256").expect("independently pinned SHA256SUMS hash");
    assert_eq!(pin.len(), 64);
    assert!(pin.bytes().all(|b| b.is_ascii_hexdigit()));
    let hash = Command::new("sha256sum")
        .arg(root.join("SHA256SUMS"))
        .output()
        .unwrap();
    assert!(hash.status.success());
    assert_eq!(
        String::from_utf8(hash.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        pin
    );
    // Require every consumed file in the authenticated manifest, not merely a
    // successful check of an unrelated subset. Filenames are fixed by this gate.
    let sums = std::fs::read_to_string(root.join("SHA256SUMS")).unwrap();
    let files = sums
        .lines()
        .map(|l| l.split_whitespace().nth(1).unwrap().trim_start_matches('*'))
        .collect::<std::collections::BTreeSet<_>>();
    assert!(files.contains("manifest.json"));
    for rank in 0..2 {
        for suffix in ["input.bf16", "weight.fp8", "scale.bf16", "hf.partial.bf16"] {
            assert!(files.contains(format!("rank{rank}.{suffix}").as_str()));
        }
    }
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&root)
            .status()
            .unwrap()
            .success()
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(metadata["schema"], "ferrule.mixed-routed-down.tp2.v1");
    assert_eq!(metadata["role"], "RoutedExpertDown");
    assert_eq!(metadata["tp"], 2);
    assert_eq!(metadata["rows"], 1);
    assert_eq!(metadata["oracle"], "hf-eager-bf16-rank-partial");
    assert!(
        metadata["checkpoint_sha256"]
            .as_str()
            .is_some_and(|s| s.len() == 64)
    );
    assert!(metadata["layer"].as_u64().is_some());
    assert!(metadata["expert"].as_u64().is_some());
    let n = usize::try_from(metadata["n"].as_u64().unwrap()).unwrap();
    let k = usize::try_from(metadata["local_k"].as_u64().unwrap()).unwrap();
    assert!((128..=8192).contains(&n) && n.is_multiple_of(128));
    assert!((128..=4096).contains(&k) && k.is_multiple_of(128));
    let ranks = (0..2)
        .map(|rank| {
            let read = |suffix: &str, len: usize| {
                let path = root.join(format!("rank{rank}.{suffix}"));
                assert_eq!(std::fs::metadata(&path).unwrap().len(), len as u64);
                std::fs::read(path).unwrap()
            };
            let input = words(&read("input.bf16", k * 2));
            let partial = words(&read("hf.partial.bf16", n * 2));
            assert!(input.iter().chain(&partial).all(|w| w & 0x7f80 != 0x7f80));
            let weight = read("weight.fp8", n * k);
            let scales = read("scale.bf16", (n / 128) * (k / 128) * 2);
            ferrule_backend::cuda_plan::NumericFp8Layout {
                n,
                k,
                row_origin: 0,
                column_origin: 0,
                scale_type: ferrule_backend::cuda_plan::NumericFp8ScaleType::Bf16,
            }
            .validate_payload(&weight, &scales)
            .unwrap();
            Rank {
                input,
                weight,
                scales,
                partial,
            }
        })
        .collect();
    Frozen { n, k, ranks }
}

#[test]
#[ignore = "CPU only; requires independently pinned real TP2 expert-down inputs/raw FP8/BF16 scales/partials"]
fn frozen_tp2_down_fixture_preflight() {
    let fixture = load();
    assert_eq!(fixture.ranks.len(), 2);
    for rank in fixture.ranks {
        assert_eq!(rank.input.len(), fixture.k);
        assert_eq!(rank.weight.len(), fixture.n * fixture.k);
        assert_eq!(rank.scales.len(), (fixture.n / 128) * (fixture.k / 128) * 2);
        assert_eq!(rank.partial.len(), fixture.n);
    }
}

#[cfg(feature = "cuda")]
fn exact_report(lane: usize, label: &str, actual: &[u16], expected: &[u16]) -> serde_json::Value {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0_f32;
    let mut mismatches = Vec::new();
    for (column, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let av = f32::from_bits(u32::from(a) << 16);
        let ev = f32::from_bits(u32::from(e) << 16);
        if av.is_finite() && ev.is_finite() {
            max_abs = max_abs.max((av - ev).abs());
        }
        if a != e {
            mismatches.push(serde_json::json!({
                "row": 0, "column": column, "actual_word": a, "expected_word": e,
                "actual": av, "expected": ev,
            }));
        }
    }
    let report = serde_json::json!({
        "rank": lane, "comparison": label, "words": actual.len(),
        "exact": mismatches.is_empty(), "mismatch_count": mismatches.len(),
        "max_abs_finite": max_abs, "mismatches": mismatches,
        "nonfinite_actual": actual.iter().filter(|w| **w & 0x7f80 == 0x7f80).count(),
    });
    println!("EXACT_REPORT {report}");
    report
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "sole TP2 GPU owner after CPU gates; exact frozen rank words, no threshold or model run"]
fn frozen_tp2_down_w8a16_and_native_match_every_word() {
    use ferrule_backend::cuda::operators::linear::*;
    use ferrule_model::execution::bf16_rne_word;
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let f = load(); // Authentication and bounded payload checks precede any CUDA owner.
    let owners = [
        CudaOperators::new_on_device(0).unwrap(),
        CudaOperators::new_on_device(1).unwrap(),
    ];
    let mut reports = Vec::new();
    for (lane, op) in owners.iter().enumerate() {
        let rank = &f.ranks[lane];
        let layout = NumericFp8Layout {
            n: f.n,
            k: f.k,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        };
        let artifact = op
            .upload_numeric_fp8_linear(layout, &rank.weight, &rank.scales)
            .unwrap();
        let input = op.upload_bf16_words(&rank.input).unwrap();
        let plan =
            NumericFp8LinearPlan::new(layout, 1, 0, NumericFp8Precision::W8A16Bf16F32Accumulate)
                .unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        let mut accumulator = op.upload_f32_buffer(&vec![7331.0; f.n + 8]).unwrap();
        // Wrong stride must fail before touching output, without poisoning scratch.
        assert!(
            op.numeric_fp8_bf16_linear_into(
                &artifact,
                &input,
                &mut accumulator,
                &mut workspace,
                plan,
                f.k - 1,
                f.n
            )
            .is_err()
        );
        assert!(!workspace.is_poisoned());
        assert_eq!(
            op.download_f32_buffer(&accumulator).unwrap(),
            vec![7331.0; f.n + 8]
        );
        op.numeric_fp8_bf16_linear_into(
            &artifact,
            &input,
            &mut accumulator,
            &mut workspace,
            plan,
            f.k,
            f.n,
        )
        .unwrap();
        let output = op
            .f32_rows_to_bf16_rne(
                &accumulator,
                F32ToBf16RowsLayout::packed(1, 1, f.n).unwrap(),
            )
            .unwrap();
        let actual = op
            .download_bf16_buffer(&output)
            .unwrap()
            .into_iter()
            .map(bf16_rne_word)
            .collect::<Vec<_>>();
        assert_eq!(
            &op.download_f32_buffer(&accumulator).unwrap()[f.n..],
            &[7331.0; 8]
        );
        assert_eq!(
            op.download_bf16_buffer(&input)
                .unwrap()
                .into_iter()
                .map(bf16_rne_word)
                .collect::<Vec<_>>(),
            rank.input
        );
        let native_plan = Bf16GemmExPlan::new(layout, 1, 64 << 20, 16 << 20).unwrap();
        let mut native = op.prepare_bf16_gemmex(native_plan).unwrap();
        let mut native_output = op.zero_bf16_buffer(f.n).unwrap();
        native
            .execute_view(op, native_plan, &artifact, &input, &mut native_output)
            .unwrap();
        let native_words = op
            .download_bf16_buffer(&native_output)
            .unwrap()
            .into_iter()
            .map(bf16_rne_word)
            .collect::<Vec<_>>();
        reports.push(exact_report(
            lane,
            "native-vs-hf",
            &native_words,
            &rank.partial,
        ));
        reports.push(exact_report(lane, "w8-vs-hf", &actual, &rank.partial));
        if let Some(directory) = std::env::var_os("FERRULE_MIXED_DOWN_REPORT_DIR") {
            let directory = PathBuf::from(directory);
            for (label, values) in [("native", &native_words), ("w8", &actual)] {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(directory.join(format!("rank{lane}.{label}.bf16")))
                    .unwrap();
                for word in values {
                    file.write_all(&word.to_le_bytes()).unwrap();
                }
            }
        }
    }
    if let Some(directory) = std::env::var_os("FERRULE_MIXED_DOWN_REPORT_DIR") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(PathBuf::from(directory).join("exact.json"))
            .unwrap();
        file.write_all(&serde_json::to_vec_pretty(&reports).unwrap())
            .unwrap();
    }
    assert!(
        reports.iter().all(|r| r["exact"] == true),
        "exact BF16 gate failed; native and W8 evidence retained for both ranks"
    );
}

#[test]
#[ignore = "CPU-only real checkpoint headers, no payload materialization; FERRULE_MIXED_HEADER_DIR required"]
fn real_tp2_candidate_header_budget_stays_within_22gib() {
    use ferrule_common::{ParallelRankId, ParallelTopologyId};
    use ferrule_model::transformer::{hybrid_parallel::*, *};
    use std::sync::Arc;
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let path = PathBuf::from(std::env::var_os("FERRULE_MIXED_HEADER_DIR").unwrap());
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(&path).unwrap();
    let layers = resources.spec().layers().len();
    assert_eq!(layers, 40);
    let topology = HybridParallelPlan::new_with_execution(
        resources.spec(),
        0..layers,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        vec![ParallelRankId::new(0), ParallelRankId::new(1)],
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    for lane in 0..2 {
        let plan = HybridMixedRankProgramPlan::for_segment(
            Arc::new(topology.rank_layout(lane).unwrap()),
            LayerSegmentPlan::new(layers, 0..layers, true, true).unwrap(),
            8,
            2,
            128,
            16,
            5,
        )
        .unwrap();
        let native = plan.operation_estimate(&resources, 64).unwrap();
        let candidate = plan
            .operation_estimate_with_routed_down_m1_candidate(&resources, 64)
            .unwrap();
        assert!(candidate.operation_grants > native.operation_grants);
        let mut without_accumulators = candidate;
        without_accumulators.operation_grants = native.operation_grants;
        assert_eq!(
            without_accumulators, native,
            "do not subtract fallback or source-retirement reserves"
        );
        let total = candidate
            .require_device_capacity(22 << 30, 128 << 20)
            .unwrap();
        assert!(
            candidate
                .require_device_capacity(total - 1, 128 << 20)
                .is_err()
        );
        assert_eq!(
            candidate.require_device_capacity(total, 128 << 20).unwrap(),
            total
        );
        eprintln!(
            "candidate rank={lane} extra_accumulator_grants={} complete_with_128m_margin={total} cap={}",
            candidate.operation_grants - native.operation_grants,
            22usize << 30
        );
    }
}
