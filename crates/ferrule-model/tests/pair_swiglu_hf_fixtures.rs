//! Independent frozen HF activations; no production/oracle arithmetic is duplicated.
#![cfg(feature = "cuda")]

use ferrule_backend::{
    cuda::operators::linear::{Bf16GemmExPlan, Bf16GemmLayout, CudaBf16Buffer, CudaOperators},
    cuda_plan::{NumericFp8Layout, NumericFp8ScaleType},
};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

const RELEASE: &str = "807e5b5643ecce603088052e39fbee36ce15dca4a3ba319b51be120e2922d40c";
const N: usize = 256;
const K: usize = 2048;

fn hash(path: &Path, expected: &str) {
    let output = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .next(),
        Some(expected),
        "fixture hash: {}",
        path.display()
    );
}
fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
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
fn download(op: &CudaOperators, buffer: &CudaBf16Buffer) -> Vec<u16> {
    buffer
        .as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap()
}
fn write_new(path: &Path, bytes: &[u8]) {
    use std::io::Write;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}
fn tensor(dir: &Path, manifest: &Value, name: &str, dtype: &str, shape: [usize; 2]) -> Vec<u8> {
    let metadata = &manifest["files"][name];
    assert_eq!(metadata["dtype"], dtype);
    assert_eq!(metadata["shape"], json!(shape));
    let bytes_per_element = if dtype == "BF16" { 2 } else { 1 };
    let bytes = shape[0] * shape[1] * bytes_per_element;
    assert_eq!(metadata["bytes"], bytes);
    let path = dir.join(name);
    hash(&path, metadata["sha256"].as_str().unwrap());
    let data = fs::read(path).unwrap();
    assert_eq!(data.len(), bytes);
    data
}

#[test]
#[ignore = "sole GPU owner: ten pinned real HF M1/M5 fixtures, cold FP8 and resident BF16, every u16"]
fn real_hf_pair_swiglu_cold_and_resident_every_word() {
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let root = std::path::PathBuf::from(std::env::var("FERRULE_PAIR_SWIGLU_FIXTURE_ROOT").unwrap());
    let evidence = std::path::PathBuf::from(std::env::var("FERRULE_MIXED_EVIDENCE_DIR").unwrap());
    hash(&root.join("manifest.json"), RELEASE);
    let release = read_json(&root.join("manifest.json"));
    assert_eq!(release["status"], "complete");
    assert_eq!(release["fixture_ready"], true);
    let fixtures = release["fixtures"].as_array().unwrap();
    assert_eq!(fixtures.len(), 10);
    // Complete admission precedes device creation/submission.
    let admitted: Vec<_> = fixtures
        .iter()
        .map(|entry| {
            let name = entry["directory"].as_str().unwrap();
            assert!(!name.contains('/') && !name.contains(".."));
            let dir = root.join(name);
            hash(&dir.join("SHA256SUMS"), entry["sha256"].as_str().unwrap());
            let checks = Command::new("sha256sum")
                .args(["--strict", "-c", "SHA256SUMS"])
                .current_dir(&dir)
                .output()
                .unwrap();
            assert!(
                checks.status.success(),
                "{}: {}",
                name,
                String::from_utf8_lossy(&checks.stderr)
            );
            let manifest = read_json(&dir.join("manifest.json"));
            assert_eq!(manifest["fixture_ready"], true);
            assert_eq!(manifest["tp"], 2);
            for field in ["rank", "rows", "n", "k", "layer", "expert"] {
                assert_eq!(manifest[field], entry[field]);
            }
            let m = manifest["rows"].as_u64().unwrap() as usize;
            let rank = manifest["rank"].as_u64().unwrap() as usize;
            assert!(matches!(m, 1 | 5) && rank < 2);
            assert_eq!(manifest["n"], N);
            assert_eq!(manifest["k"], K);
            assert_eq!(manifest["packed_n"], 2 * N);
            assert_eq!(
                manifest["gate_input_token_indices"],
                if m == 5 {
                    json!([1, 2, 4, 3, 0])
                } else {
                    json!([0])
                }
            );
            assert_eq!(
                manifest["normalized_hidden_token_indices"],
                json!((0..m).collect::<Vec<_>>())
            );
            let input = words(&tensor(&dir, &manifest, "gateupinput.bf16", "BF16", [m, K]));
            let normalized = words(&tensor(
                &dir,
                &manifest,
                "normalized_hidden.bf16",
                "BF16",
                [m, K],
            ));
            let order = manifest["gate_input_token_indices"].as_array().unwrap();
            for (row, index) in order.iter().enumerate() {
                let index = index.as_u64().unwrap() as usize;
                assert_eq!(
                    &input[row * K..(row + 1) * K],
                    &normalized[index * K..(index + 1) * K]
                );
            }
            let expected = words(&tensor(
                &dir,
                &manifest,
                "hf.activation.bf16",
                "BF16",
                [m, N],
            ));
            assert!(
                expected.iter().all(|w| w & 0x7f80 != 0x7f80),
                "finite HF contract"
            );
            let gate = tensor(&dir, &manifest, "gate.weight.fp8", "F8_E4M3", [N, K]);
            let up = tensor(&dir, &manifest, "up.weight.fp8", "F8_E4M3", [N, K]);
            let gate_scale = tensor(
                &dir,
                &manifest,
                "gate.scale.bf16",
                "BF16",
                [N / 128, K / 128],
            );
            let up_scale = tensor(&dir, &manifest, "up.scale.bf16", "BF16", [N / 128, K / 128]);
            for part in ["gate", "up"] {
                let source = &manifest["files"][format!("{part}.scale.bf16")]["source"];
                assert_eq!(source["original"]["original_header"]["dtype"], "BF16");
                assert_eq!(source["global_rows"], json!([rank * 2, (rank + 1) * 2]));
            }
            (
                name.to_owned(),
                m,
                rank,
                input,
                expected,
                gate,
                up,
                gate_scale,
                up_scale,
            )
        })
        .collect();
    assert_eq!(admitted.iter().filter(|f| f.1 == 1).count(), 8);
    assert_eq!(admitted.iter().filter(|f| f.1 == 5).count(), 2);
    let layout = NumericFp8Layout {
        n: N,
        k: K,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    };
    let op = CudaOperators::new_on_device(0).unwrap();
    let plans: Vec<_> = [1, 5]
        .into_iter()
        .map(|m| {
            Bf16GemmExPlan::new(
                NumericFp8Layout { n: 2 * N, ..layout },
                m,
                128 << 20,
                16 << 20,
            )
            .unwrap()
        })
        .collect();
    let mut owner = op
        .prepare_bf16_gemmex_views(&plans, plans[1].storage_grant_bytes(), 16 << 20)
        .unwrap();
    let mut results = Vec::new();
    let mut failures = 0usize;
    for (name, m, rank, input, expected, gate, up, gate_scale, up_scale) in admitted {
        layout.validate_payload(&gate, &gate_scale).unwrap();
        layout.validate_payload(&up, &up_scale).unwrap();
        let gate = op
            .upload_numeric_fp8_linear(layout, &gate, &gate_scale)
            .unwrap();
        let up = op
            .upload_numeric_fp8_linear(layout, &up, &up_scale)
            .unwrap();
        let input = op.upload_bf16_words(&input).unwrap();
        let plan = plans[usize::from(m == 5)];
        let mut cold = op.zero_bf16_buffer(m * N).unwrap();
        owner
            .execute_pair_swiglu_view_into(&op, plan, [&gate, &up], &input, &mut cold)
            .unwrap();
        let mut decoded = op.zero_bf16_buffer(2 * N * K).unwrap();
        owner
            .decode_numeric_bf16_pair_into(&op, [&gate, &up], &mut decoded)
            .unwrap();
        let mut resident = op.zero_bf16_buffer(m * N).unwrap();
        owner
            .execute_bf16_pair_swiglu_view_into(
                &op,
                plan,
                &decoded,
                &input,
                &mut resident,
                Bf16GemmLayout::contiguous(m, 2 * N, K),
            )
            .unwrap();
        for (path, actual) in [
            ("cold", download(&op, &cold)),
            ("resident", download(&op, &resident)),
        ] {
            assert_eq!(actual.len(), expected.len());
            let mismatch: Vec<_> = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(index, (a, b))| json!({"index":index,"actual":a,"expected":b}))
                .collect();
            failures += mismatch.len();
            let bytes: Vec<_> = actual.iter().flat_map(|w| w.to_le_bytes()).collect();
            write_new(&evidence.join(format!("{name}.{path}.actual.bf16")), &bytes);
            println!(
                "HF_PAIR_SWIGLU fixture={name} rank={rank} M={m} path={path} words={} mismatches={} first={:?}",
                actual.len(),
                mismatch.len(),
                mismatch.first()
            );
            results.push(json!({"fixture":name,"rank":rank,"rows":m,"path":path,"words":actual.len(),"mismatches":mismatch}));
        }
    }
    write_new(&evidence.join("hf-pair-swiglu-result.json"), &serde_json::to_vec_pretty(&json!({"release_sha256":RELEASE,"fixtures":10,"entries":20,"mismatching_words":failures,"passed":failures==0,"results":results})).unwrap());
    assert_eq!(
        failures, 0,
        "independent HF BF16 word gate failed; do not run E2E or relax exactness"
    );
}
