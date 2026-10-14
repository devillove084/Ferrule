//! Independent F32 and explicit BF16/profile readers with separate comparison bounds.
//! Shared by resident EP and PP/TP factory tests; never compiled into production.
#[cfg(feature = "cuda")]
use ferrule_model::{CheckpointDType, CheckpointTensorReader, CheckpointTensorSlice, TensorRole};
use serde_json::Value;
#[cfg(feature = "cuda")]
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

pub(in crate::model_factory) const ATOL: f64 = 2e-4;
pub(in crate::model_factory) const RTOL: f64 = 2e-4;

#[cfg(feature = "cuda")]
pub(in crate::model_factory) struct Oracle {
    path: PathBuf,
    pub(in crate::model_factory) header: Value,
    data_start: u64,
    model_bf16: bool,
    pub(in crate::model_factory) report: Value,
}

#[cfg(feature = "cuda")]
impl Oracle {
    pub(in crate::model_factory) fn open(directory: &Path, case: &str) -> Self {
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["schema"], "ferrule.qwen35-fp8-reference.f32.v1");
        assert_eq!(manifest["status"], "complete");
        assert_eq!(manifest["full_reference_complete"], true);
        assert_eq!(manifest["execution"]["precision_profile"], "f32");
        assert_eq!(manifest["execution"]["tf32"], false);
        Self::open_manifest(directory, case, &manifest, false)
    }

    pub(in crate::model_factory) fn open_bf16(directory: &Path, case: &str, pin: &str) -> Self {
        check_hash(&directory.join("manifest.json"), pin);
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        validate_bf16_manifest(&manifest);
        for key in ["source_payloads", "loading_evidence"] {
            let item = &manifest[key];
            check_hash(
                &directory.join(item["file"].as_str().unwrap()),
                item["sha256"].as_str().unwrap(),
            );
        }
        Self::open_manifest(directory, case, &manifest, true)
    }

    pub(in crate::model_factory) fn open_bf16_fresh(
        directory: &Path,
        case: &str,
        pin: &str,
    ) -> Self {
        check_hash(&directory.join("manifest.json"), pin);
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest["schema"],
            "ferrule.bf16-fresh-quality-reference.v1"
        );
        assert_eq!(manifest["status"], "complete");
        let original = Path::new(
            manifest["original_reference"]["directory"]
                .as_str()
                .unwrap(),
        );
        check_hash(
            &original.join("manifest.json"),
            manifest["original_reference"]["sha256"].as_str().unwrap(),
        );
        let baseline: Value =
            serde_json::from_slice(&std::fs::read(original.join("manifest.json")).unwrap())
                .unwrap();
        validate_bf16_manifest(&baseline);
        for field in [
            "profile",
            "router",
            "native_source",
            "versions",
            "precision_contract",
        ] {
            assert_eq!(manifest[field], baseline[field], "fresh oracle {field}");
        }
        let source = &manifest["source_payloads"];
        check_hash(
            &directory.join(source["file"].as_str().unwrap()),
            source["sha256"].as_str().unwrap(),
        );
        Self::open_manifest(directory, case, &manifest, true)
    }

    fn open_manifest(directory: &Path, case: &str, manifest: &Value, model_bf16: bool) -> Self {
        let entry = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == case)
            .unwrap();
        if model_bf16 {
            check_hash(
                &directory.join(entry["report"].as_str().unwrap()),
                entry["report_sha256"].as_str().unwrap(),
            );
        }
        let report: Value = serde_json::from_slice(
            &std::fs::read(directory.join(entry["report"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        assert_eq!(report["sha256"], entry["sha256"]);
        assert_eq!(
            report["calls"].as_array().unwrap().len(),
            if manifest["schema"] == "ferrule.bf16-fresh-quality-reference.v1" {
                9
            } else {
                2
            }
        );
        let path = directory.join(report["file"].as_str().unwrap());
        let checksum = Command::new("sha256sum").arg(&path).output().unwrap();
        assert!(checksum.status.success());
        assert_eq!(
            String::from_utf8(checksum.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            entry["sha256"].as_str().unwrap()
        );
        let mut file = std::fs::File::open(&path).unwrap();
        let mut len = [0; 8];
        file.read_exact(&mut len).unwrap();
        let len = u64::from_le_bytes(len);
        assert!(len < 16 * 1024 * 1024);
        let mut header = vec![0; len as usize];
        file.read_exact(&mut header).unwrap();
        let header: Value = serde_json::from_slice(&header).unwrap();
        if model_bf16 {
            assert_eq!(header["__metadata__"]["schema"], BF16_SCHEMA);
            assert_eq!(header["__metadata__"]["profile"], BF16_PROFILE);
            assert_eq!(header["__metadata__"]["model_boundary"], "bf16");
            assert_eq!(
                header["__metadata__"]["router_tie_policy"],
                "lower_expert_id"
            );
            assert_eq!(report["router"], manifest["router"]);
            assert_eq!(report["profile"], BF16_PROFILE);
        }
        Self {
            path,
            header,
            model_bf16,
            data_start: 8 + len,
            report,
        }
    }

    pub(in crate::model_factory) fn values(&self, name: &str) -> Vec<f32> {
        let tensor = &self.header[name];
        let bf16 = self.model_bf16
            && !name.contains(".recurrent_states.")
            && !name.ends_with("routing_weights")
            && !name.ends_with("router_probabilities");
        assert_eq!(tensor["dtype"], if bf16 { "BF16" } else { "F32" });
        let start = tensor["data_offsets"][0].as_u64().unwrap();
        let end = tensor["data_offsets"][1].as_u64().unwrap();
        let shape = serde_json::from_value(tensor["shape"].clone()).unwrap();
        let bytes = CheckpointTensorReader::new(64 * 1024 * 1024)
            .read_slice(&CheckpointTensorSlice {
                name: name.into(),
                role: TensorRole::Unknown,
                path: self.path.clone(),
                offset: self.data_start + start,
                bytes: end - start,
                dtype: if bf16 {
                    CheckpointDType::Bf16
                } else {
                    CheckpointDType::F32
                },
                shape,
            })
            .unwrap()
            .bytes;
        if bf16 {
            assert!(bytes.len().is_multiple_of(2));
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect()
        } else {
            assert!(bytes.len().is_multiple_of(4));
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        }
    }

    pub(in crate::model_factory) fn ids(&self, name: &str) -> Vec<usize> {
        let tensor = &self.header[name];
        assert_eq!(tensor["dtype"], "I64");
        let start = tensor["data_offsets"][0].as_u64().unwrap();
        let end = tensor["data_offsets"][1].as_u64().unwrap();
        let shape = serde_json::from_value(tensor["shape"].clone()).unwrap();
        CheckpointTensorReader::new(16 * 1024 * 1024)
            .read_slice(&CheckpointTensorSlice {
                name: name.into(),
                role: TensorRole::Unknown,
                path: self.path.clone(),
                offset: self.data_start + start,
                bytes: end - start,
                dtype: CheckpointDType::I64,
                shape,
            })
            .unwrap()
            .bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| usize::try_from(i64::from_le_bytes(*bytes)).unwrap())
            .collect()
    }
}

#[derive(Default)]
pub(in crate::model_factory) struct StrictReport {
    pub(in crate::model_factory) metrics: Vec<Value>,
    pub(in crate::model_factory) failures: Vec<String>,
    pub(in crate::model_factory) logits: usize,
}
impl StrictReport {
    pub(in crate::model_factory) fn route_ids(
        &mut self,
        actual: &[usize],
        expected: &[usize],
        label: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} shape");
        assert!(!actual.is_empty(), "{label} empty routes");
        let mismatches = actual.iter().zip(expected).filter(|(a, e)| a != e).count();
        if mismatches > 0 {
            eprintln!("FIRST ROUTE DIVERGENCE {label}: actual={actual:?} expected={expected:?}");
            self.failures.push(label.into());
        }
        // Preserve original rank-slot order, never compare sorted sets.
        self.metrics
            .push(serde_json::json!({"label":label, "elements":actual.len(),
            "failures":mismatches, "actual_ids":actual, "expected_ids":expected}));
    }
    pub(in crate::model_factory) fn values(
        &mut self,
        actual: &[f32],
        expected: &[f32],
        label: &str,
    ) {
        self.values_with_bounds(actual, expected, label, ATOL, RTOL);
    }
    pub(in crate::model_factory) fn bf16_values(
        &mut self,
        actual: &[f32],
        expected: &[f32],
        label: &str,
    ) {
        self.values_with_bounds(actual, expected, label, BF16_ATOL, BF16_RTOL);
    }
    fn values_with_bounds(
        &mut self,
        actual: &[f32],
        expected: &[f32],
        label: &str,
        atol: f64,
        rtol: f64,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} shape");
        assert!(!actual.is_empty(), "{label} empty comparison");
        let mut max_abs = 0.0f64;
        let mut max_tolerance_ratio = 0.0f64;
        let mut sum_squared = 0.0f64;
        let mut failures = 0;
        let mut nonfinite = 0;
        let mut first_failure = None;
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let finite = a.is_finite() && e.is_finite();
            let diff = (f64::from(a) - f64::from(e)).abs();
            let tolerance = atol + rtol * f64::from(e).abs();
            if finite {
                max_abs = max_abs.max(diff);
                max_tolerance_ratio = max_tolerance_ratio.max(diff / tolerance);
                sum_squared += diff * diff;
            } else {
                nonfinite += 1;
            }
            if !finite || diff > tolerance {
                failures += 1;
                first_failure.get_or_insert(serde_json::json!({"index":index,"actual":a,"expected":e,"abs_error":diff,"tolerance":tolerance}));
            }
        }
        let rms = (sum_squared / actual.len() as f64).sqrt();
        eprintln!(
            "{label}: n={} max_abs={max_abs:.9e} rms={rms:.9e} max_tolerance_ratio={max_tolerance_ratio:.9e} nonfinite={nonfinite} failures={failures} first={first_failure:?}",
            actual.len()
        );
        self.metrics.push(serde_json::json!({"label":label,"elements":actual.len(),"max_abs":max_abs,"rms":rms,
            "atol":atol,"rtol":rtol,"max_tolerance_ratio":max_tolerance_ratio,"nonfinite":nonfinite,"failures":failures,"first_failure":first_failure}));
        if failures > 0 {
            self.failures.push(label.into());
        }
    }
}

#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn compare_routes(
    report: &mut StrictReport,
    routes: &BTreeMap<usize, (usize, Vec<usize>, Vec<f32>)>,
    oracle: &Oracle,
    case: &str,
    stage: &str,
    rows: usize,
) {
    assert_eq!(routes.len(), 40, "{stage} route layer count");
    for (layer, (actual_rows, actual_ids, actual_weights)) in routes {
        assert_eq!(*actual_rows, rows);
        let prefix = format!("{stage}.layers.{layer:02}");
        let expected_ids = oracle.ids(&format!("{prefix}.selected_expert_ids"));
        let expected_weights = oracle.values(&format!("{prefix}.routing_weights"));
        assert_eq!(actual_ids.len(), rows * 8);
        assert_eq!(expected_ids.len(), actual_ids.len());
        assert_eq!(actual_weights.len(), rows * 8);
        report.route_ids(
            actual_ids,
            &expected_ids,
            &format!("{case}/{prefix} ordered route IDs"),
        );
        report.values(
            actual_weights,
            &expected_weights,
            &format!("{case}/{prefix} routing_weights"),
        );
    }
}

#[test]
fn strict_comparator_preserves_original_f32_bounds_and_counts_every_element() {
    let mut report = StrictReport::default();
    report.values(&[0., 1., -1.], &[0., 1., -1.], "equal");
    report.values(&[0.00019, 1.00039], &[0., 1.], "inside");
    assert!(report.failures.is_empty());
    report.values(&[0.00021, 1.00041], &[0., 1.], "outside");
    assert_eq!(report.failures, ["outside"]);
    assert_eq!(report.metrics[2]["elements"], 2);
    assert_eq!(report.metrics[2]["failures"], 2);
    report.logits += 2;
    assert_eq!(report.logits, 2);
}

#[test]
fn strict_comparator_rejects_nonfinite_either_operand_and_shape_mismatch() {
    let mut report = StrictReport::default();
    report.values(
        &[f32::NAN, 1., f32::INFINITY],
        &[0., f32::NEG_INFINITY, 1.],
        "nonfinite",
    );
    assert_eq!(report.metrics[0]["nonfinite"], 3);
    assert_eq!(report.metrics[0]["failures"], 3);
    assert_eq!(report.failures, ["nonfinite"]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| report.values(
            &[0.],
            &[0., 0.],
            "shape"
        )))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| report.values(
            &[],
            &[],
            "empty"
        )))
        .is_err()
    );
}

#[test]
fn strict_route_comparator_preserves_rank_slot_order() {
    let mut report = StrictReport::default();
    report.route_ids(&[3, 1, 2], &[3, 1, 2], "equal");
    assert!(report.failures.is_empty());
    report.route_ids(&[1, 3, 2], &[3, 1, 2], "reordered");
    assert_eq!(report.failures, ["reordered"]);
    assert_eq!(report.metrics[1]["failures"], 2);
    assert_eq!(
        report.metrics[1]["actual_ids"],
        serde_json::json!([1, 3, 2])
    );
}

// The existing BF16 hybrid tiny policy, not a relaxed full40-specific bound.
pub(in crate::model_factory) const BF16_ATOL: f64 = 0.00025;
pub(in crate::model_factory) const BF16_RTOL: f64 = 0.012;
pub(in crate::model_factory) const BF16_SCHEMA: &str =
    "ferrule.qwen35-fp8-profile-bf16-cuda-reference.v1";
pub(in crate::model_factory) const BF16_PROFILE: &str =
    "cuda-hybrid-numeric-fp8-model-bf16-f32-ssm";

#[cfg(feature = "cuda")]
pub(in crate::model_factory) fn check_hash(path: &Path, pin: &str) {
    assert_eq!(pin.len(), 64, "explicit SHA256 required");
    let result = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(result.status.success());
    assert_eq!(
        String::from_utf8(result.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        pin
    );
}

pub(in crate::model_factory) fn validate_bf16_manifest(m: &Value) {
    assert_eq!(m["schema"], BF16_SCHEMA);
    assert_eq!(m["profile"], BF16_PROFILE);
    assert_eq!(m["status"], "complete");
    assert_eq!(m["full_reference_complete"], true);
    assert_eq!(m["reference_kind"], "independent_profile_bf16_full_model");
    assert_eq!(m["execution"]["device"], "cuda");
    assert_eq!(m["execution"]["model_boundary"], "bf16");
    assert_eq!(m["execution"]["tf32"], false);
    assert_eq!(m["execution"]["production_imports"], false);
    assert_eq!(m["execution"]["reduction_emulation"], false);
    assert_eq!(m["precision_contract"]["profile"], BF16_PROFILE);
    assert_eq!(m["precision_contract"]["accumulation"], "F32");
    assert_eq!(m["precision_contract"]["fp8"], "E4M3FN");
    assert_eq!(
        m["precision_contract"]["scale"],
        "original numeric BF16 multiplicative 128x128"
    );
    assert_eq!(m["precision_contract"]["tf32"], false);
    assert_eq!(
        m["precision_contract"]["bf16_reduced_precision_reduction"],
        false
    );
    assert_eq!(m["arguments"]["model_boundary"], "bf16");
    assert_eq!(m["arguments"]["execute_cuda"], true);
    assert_eq!(m["versions"]["torch"], "2.13.0+cu129");
    assert_eq!(m["versions"]["transformers"], "5.5.3");
    assert_eq!(
        m["native_source"]["modeling_qwen3_5_moe"]["sha256"],
        "0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647"
    );
    assert_eq!(m["router"]["policy"], "lower_expert_id");
    assert_eq!(m["router"]["native_policy"], "native_unstable");
    assert_eq!(m["router"]["canonicalization"], "equal_raw_logits_only");
    assert_eq!(m["router"]["unmodified_native_router"], false);
    assert_eq!(
        m["router"]["weights"],
        "native F32 selected renormalized weights unchanged"
    );
    assert_eq!(
        m["router"]["unique_raw_logits"],
        "native selection unchanged, including softmax collisions"
    );
    assert_eq!(
        m["router"]["probabilities"],
        "native F32 full softmax unchanged"
    );
    assert_eq!(
        m["native_source"]["cache_utils"]["sha256"],
        "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586"
    );
    assert_eq!(
        m["native_source"]["modeling_utils"]["sha256"],
        "b8467e1ada952862d2e4d76632475ac2f9fa198121d0e1ce64fa393ea3773e16"
    );
    assert_eq!(m["arguments"]["ssm_policy"], "f32");
    assert_eq!(m["arguments"]["router_tie_policy"], "lower_expert_id");
    assert_eq!(m["precision_contract"]["model_boundary"], "bf16");
    for field in ["activation", "output", "kv", "conv"] {
        assert_eq!(m["precision_contract"][field], "BF16");
    }
    assert_eq!(
        m["precision_contract"]["ssm"],
        "F32 explicitly preallocated; NOT the HF default cache dtype"
    );
    let c = &m["checkpoint"]["effective_text_config"];
    assert_eq!(c["num_hidden_layers"], 40);
    assert_eq!(c["num_experts"], 256);
    assert_eq!(c["num_experts_per_tok"], 8);
    assert_eq!(c["vocab_size"], 248320);
    assert_eq!(
        m["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["hello", "capital"]
    );
}

#[test]
fn bf16_comparator_uses_tiny_bounds_without_changing_f32_gate() {
    let mut report = StrictReport::default();
    report.bf16_values(&[0.00024, 1.01], &[0., 1.], "bf16 inside");
    assert!(report.failures.is_empty());
    report.bf16_values(&[0.00026, 1.013], &[0., 1.], "bf16 outside");
    assert_eq!(report.metrics[1]["failures"], 2);
    report.values(&[1.01], &[1.], "old f32 still strict");
    assert_eq!(report.metrics[2]["atol"], ATOL);
    assert_eq!(report.metrics[2]["rtol"], RTOL);
    assert_eq!(report.metrics[2]["failures"], 1);
}

#[test]
fn bf16_manifest_rejects_unknown_policy_native_f32_and_incomplete_reference() {
    // Metadata contract only: this is deliberately not numerical reference data.
    let good = serde_json::json!({
        "schema": BF16_SCHEMA, "profile": BF16_PROFILE, "status":"complete",
        "full_reference_complete":true, "reference_kind":"independent_profile_bf16_full_model",
        "execution":{"device":"cuda","model_boundary":"bf16","tf32":false,"production_imports":false,"reduction_emulation":false},
        "versions":{"torch":"2.13.0+cu129","transformers":"5.5.3"},
        "native_source":{
            "modeling_qwen3_5_moe":{"sha256":"0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647"},
            "cache_utils":{"sha256":"a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586"},
            "modeling_utils":{"sha256":"b8467e1ada952862d2e4d76632475ac2f9fa198121d0e1ce64fa393ea3773e16"}},
        "router":{"policy":"lower_expert_id","native_policy":"native_unstable","canonicalization":"equal_raw_logits_only",
            "unmodified_native_router":false,"weights":"native F32 selected renormalized weights unchanged",
            "unique_raw_logits":"native selection unchanged, including softmax collisions","probabilities":"native F32 full softmax unchanged"},
        "arguments":{"ssm_policy":"f32","router_tie_policy":"lower_expert_id","model_boundary":"bf16","execute_cuda":true},
        "precision_contract":{"profile":BF16_PROFILE,"model_boundary":"bf16","activation":"BF16","output":"BF16","kv":"BF16","conv":"BF16",
            "ssm":"F32 explicitly preallocated; NOT the HF default cache dtype", "accumulation":"F32","fp8":"E4M3FN",
            "scale":"original numeric BF16 multiplicative 128x128","tf32":false,"bf16_reduced_precision_reduction":false},
        "checkpoint":{"effective_text_config":{"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,"vocab_size":248320}},
        "cases":[{"id":"hello"},{"id":"capital"}]
    });
    validate_bf16_manifest(&good);
    for (pointer, bad) in [
        (
            "/schema",
            serde_json::json!("ferrule.qwen35-fp8-reference.f32.v1"),
        ),
        (
            "/schema",
            serde_json::json!("ferrule.qwen35-fp8-native-bf16-cuda-reference.v1"),
        ),
        ("/status", serde_json::json!("preflight_only")),
        ("/full_reference_complete", serde_json::json!(false)),
        ("/router/policy", serde_json::json!("native_unstable")),
        (
            "/router/canonicalization",
            serde_json::json!("equal_probabilities"),
        ),
        ("/router/policy", Value::Null),
        (
            "/precision_contract/accumulation",
            serde_json::json!("BF16"),
        ),
        ("/precision_contract/kv", serde_json::json!("F32")),
        ("/arguments/ssm_policy", serde_json::json!("native")),
        ("/versions/transformers", serde_json::json!("5.2.0")),
        (
            "/checkpoint/effective_text_config/num_hidden_layers",
            serde_json::json!(4),
        ),
    ] {
        let mut invalid = good.clone();
        *invalid.pointer_mut(pointer).unwrap() = bad;
        assert!(
            std::panic::catch_unwind(|| validate_bf16_manifest(&invalid)).is_err(),
            "accepted {pointer}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn bf16_reader_preserves_words_and_cannot_relax_the_f32_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("words.bin");
    let mut bytes: Vec<u8> = [0x3f80_u16, 0xc020, 0x0001, 0x8000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    bytes.extend([0.125_f32, -1.5].into_iter().flat_map(f32::to_le_bytes));
    std::fs::write(&path, bytes).unwrap();
    let mut oracle = Oracle {
        path,
        header: serde_json::json!({
            "prefill.logits":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},
            "prefill.layers.00.routing_weights":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}
        }),
        data_start: 0,
        model_bf16: true,
        report: Value::Null,
    };
    assert_eq!(
        oracle
            .values("prefill.logits")
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [0x3f800000, 0xc0200000, 0x00010000, 0x80000000]
    );
    assert_eq!(
        oracle.values("prefill.layers.00.routing_weights"),
        [0.125, -1.5]
    );
    oracle.model_bf16 = false;
    assert!(std::panic::catch_unwind(|| oracle.values("prefill.logits")).is_err());
    assert_eq!(
        oracle.values("prefill.layers.00.routing_weights"),
        [0.125, -1.5]
    );
}
