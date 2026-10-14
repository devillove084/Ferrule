//! Independent serialized artifacts only. Never imports or runs the exporter.
use serde_json::Value;
use std::{
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::Command,
};
pub(super) fn hash(path: &Path, expected: &str) {
    assert_eq!(expected.len(), 64, "missing SHA256 pin");
    let output = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        expected,
        "artifact/source hash: {path:?}"
    );
}
pub(super) fn json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
pub(super) fn artifact(root: &Path, record: &Value) -> PathBuf {
    let path = root
        .join(record["file"].as_str().unwrap())
        .canonicalize()
        .unwrap();
    assert_eq!(path.parent().unwrap(), root.canonicalize().unwrap());
    assert!(path.metadata().unwrap().len() <= 2 << 30);
    hash(&path, record["sha256"].as_str().unwrap());
    path
}
pub struct Oracle {
    root: PathBuf,
    manifest: Value,
}
impl Oracle {
    pub fn open(root: &Path, pin: &str, model: &Path) -> Self {
        Self::open_inner(root, pin, model, false)
    }
    pub(super) fn open_lifecycle(root: &Path, pin: &str, model: &Path) -> Self {
        Self::open_inner(root, pin, model, true)
    }
    pub(super) fn manifest(&self) -> &Value {
        &self.manifest
    }
    fn open_inner(root: &Path, pin: &str, model: &Path, lifecycle: bool) -> Self {
        hash(&root.join("manifest.json"), pin);
        let m = json(&root.join("manifest.json"));
        assert_eq!(
            m["schema"],
            if lifecycle {
                "ferrule.paper.tp2-mixed-lifecycle-reference.v1"
            } else {
                "ferrule.paper.tp2-mixed-independent-oracle.v1"
            }
        );
        assert_eq!(m["profile"], "SystemMixedTP2EP1");
        assert_eq!(m["status"], "complete");
        assert_eq!(m["full_reference_complete"], true);
        assert_eq!(m["boundary_inventory_complete"], true);
        assert_eq!(m["execution"]["logical_tp"], 2);
        assert_eq!(m["execution"]["ep"], 1);
        assert_eq!(m["execution"]["pp"], 1);
        assert_eq!(m["execution"]["tf32"], false);
        assert_eq!(m["execution"]["bf16_reduced_precision_reduction"], false);
        let protocol: Value = serde_json::from_str(include_str!(
            "../../../../papers/arxiv/experiments/tp2_mixed/oracle/protocol.json"
        ))
        .unwrap();
        assert_eq!(m["protocol"], protocol);
        assert_eq!(m["versions"]["torch"], "2.13.0+cu129");
        assert_eq!(m["versions"]["transformers"], "5.5.3");
        for field in [
            "custody",
            "source_payloads",
            "operand_layout",
            "authentication",
            "numerical_authentication",
            "launch",
            "idle_before",
            "idle_after",
        ] {
            artifact(root, &m[field]);
        }
        hash(
            &model.join("config.json"),
            m["checkpoint"]["config_sha256"].as_str().unwrap(),
        );
        hash(
            &model.join("model.safetensors.index.json"),
            m["checkpoint"]["index_sha256"].as_str().unwrap(),
        );
        for (name, pin) in m["checkpoint"]["full_shards"].as_object().unwrap() {
            assert_eq!(Path::new(name).components().count(), 1);
            hash(&model.join(name), pin.as_str().unwrap());
        }
        assert!(
            !m["checkpoint"]["full_shards"]
                .as_object()
                .unwrap()
                .is_empty()
        );
        if !lifecycle {
            assert_eq!(
                m["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c["id"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                ["hello", "capital"]
            );
        } else {
            assert_eq!(m["suite"]["cases"], m["cases"]);
            assert_eq!(
                json(&artifact(root, &m["launch"]))["scenario_plan"],
                m["suite"]["plan"]
            );
        }
        Self {
            root: root.into(),
            manifest: m,
        }
    }
    pub fn case(&self, id: &str) -> Case {
        let entry = self.manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == id)
            .unwrap();
        let path = artifact(&self.root, entry);
        let report = json(&artifact(&self.root, &entry["report"]));
        artifact(&self.root, &entry["statistics"]);
        if self.manifest["schema"] == "ferrule.paper.tp2-mixed-lifecycle-reference.v1" {
            assert_eq!(report["calls"].as_array().unwrap().len(), 1);
            assert_eq!(report["id"], id);
            assert_eq!(report["sha256"], entry["sha256"]);
            assert_eq!(report["file"], entry["file"]);
            assert_eq!(report["calls"][0]["stage"], "call");
        } else {
            assert_eq!(report["calls"].as_array().unwrap().len(), 2);
            assert_eq!(report["id"], id);
            assert_eq!(report["sha256"], entry["sha256"]);
            assert_eq!(report["file"], entry["file"]);
            assert_eq!(report["calls"][0]["stage"], "prefill");
            assert_eq!(report["calls"][1]["stage"], "decode.0");
            let prompt = if id == "hello" {
                vec![9419]
            } else {
                vec![760, 6511, 314, 9338, 369]
            };
            assert_eq!(
                report["calls"][0]["input_token_ids"],
                serde_json::json!(prompt)
            );
            assert_eq!(
                report["calls"][1]["input_token_ids"],
                serde_json::json!([report["calls"][0]["next_token_id"]])
            );
            assert_eq!(report["calls"][1]["position_start"], prompt.len());
        }
        let mut file = std::fs::File::open(&path).unwrap();
        let mut n = [0; 8];
        file.read_exact(&mut n).unwrap();
        let n = u64::from_le_bytes(n);
        assert!(n < 64 << 20);
        let mut header = vec![0; n as usize];
        file.read_exact(&mut header).unwrap();
        let header: Value = serde_json::from_slice(&header).unwrap();
        // Paper save_case uses the shared tensor schema even for lifecycle manifests.
        assert_eq!(
            header["__metadata__"]["schema"],
            "ferrule.paper.tp2-mixed-independent-oracle.v1"
        );
        assert_eq!(header["__metadata__"]["profile"], self.manifest["profile"]);
        Case {
            path,
            report,
            header,
            offset: 8 + n,
        }
    }
}
pub struct Case {
    path: PathBuf,
    report: Value,
    header: Value,
    offset: u64,
}
impl Case {
    pub(super) fn call(&self, stage: &str) -> &Value {
        self.report["calls"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["stage"] == stage)
            .unwrap()
    }
    pub fn tokens(&self, stage: &str) -> Vec<u32> {
        self.call(stage)["input_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| u32::try_from(n.as_u64().unwrap()).unwrap())
            .collect()
    }
    pub fn compare(&self, stage: &str, actual: &[f32]) {
        let t = &self.header[format!("{stage}.logits")];
        assert_eq!(
            t["shape"],
            serde_json::json!([1, self.tokens(stage).len(), 248_320])
        );
        let dtype = t["dtype"].as_str().unwrap();
        let bytes = match dtype {
            "F32" => 4,
            "BF16" => 2,
            _ => panic!("unsupported oracle logit dtype"),
        };
        let start = t["data_offsets"][0].as_u64().unwrap();
        let end = t["data_offsets"][1].as_u64().unwrap();
        assert_eq!(end - start, actual.len() as u64 * bytes);
        assert!(self.offset + end <= self.path.metadata().unwrap().len());
        let mut file = std::fs::File::open(&self.path).unwrap();
        file.seek(SeekFrom::Start(self.offset + start)).unwrap();
        let mut values = vec![0; (end - start) as usize];
        file.read_exact(&mut values).unwrap();
        let expected: Vec<f32> = if bytes == 4 {
            values
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        } else {
            values
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                .collect()
        };
        let mut failures = 0usize;
        let mut nonfinite = 0usize;
        let mut max_abs = 0.0f64;
        let mut sum_squared = 0.0f64;
        for (actual, expected) in actual.iter().zip(&expected) {
            if !actual.is_finite() || !expected.is_finite() {
                nonfinite += 1;
                failures += 1;
                continue;
            }
            let delta = *actual as f64 - *expected as f64;
            let abs = delta.abs();
            max_abs = max_abs.max(abs);
            sum_squared += delta * delta;
            if abs > 0.00025 + 0.012 * (*expected as f64).abs() {
                failures += 1;
            }
        }
        let compared = actual.len();
        let finite_compared = compared - nonfinite;
        let rmse = (finite_compared > 0).then(|| (sum_squared / finite_compared as f64).sqrt());
        let greedy = actual[actual.len() - 248_320..]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        let expected_greedy = self.call(stage)["next_token_id"].as_u64().unwrap();
        let metrics = serde_json::json!({
            "case": self.report["id"], "stage": stage,
            "compared": compared, "finite_compared": finite_compared,
            "failures": failures, "nonfinite": nonfinite,
            "max_abs_finite": max_abs, "rmse_finite": rmse,
            "actual_greedy": greedy, "expected_greedy": expected_greedy,
            "atol": 0.00025, "rtol": 0.012,
        });
        eprintln!("oracle_metrics {metrics}");
        if let Some(directory) = std::env::var_os("FERRULE_MIXED_EVIDENCE_DIR") {
            use std::io::Write;
            let directory = PathBuf::from(directory).canonicalize().unwrap();
            assert_ne!(
                directory,
                self.path.parent().unwrap().canonicalize().unwrap()
            );
            let name = format!("{}.{stage}", self.report["id"].as_str().unwrap());
            let mut output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(format!("{name}.actual.f32le")))
                .unwrap();
            for value in actual {
                output.write_all(&value.to_le_bytes()).unwrap();
            }
            output.sync_all().unwrap();
            let mut report = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(format!("{name}.metrics.json")))
                .unwrap();
            serde_json::to_writer_pretty(&mut report, &metrics).unwrap();
            report.sync_all().unwrap();
        }
        assert_eq!(failures, 0, "{stage} full-vocabulary tolerance failures");
        assert_eq!(greedy as u64, expected_greedy, "greedy identity");
    }
}
