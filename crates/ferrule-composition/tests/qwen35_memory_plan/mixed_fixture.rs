//! Sparse source headers only: no weight payload, tokenizer, CUDA or owner.
use ferrule_model::models::qwen35::{Qwen35Config, Qwen35HfNameMapper, Qwen35TensorPartitionKind};
use std::io::Write;

pub struct Fixture {
    pub dir: tempfile::TempDir,
}
impl Fixture {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = include_str!("../../../ferrule-model/tests/qwen35_35b_fp8_config.json");
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        let config = Qwen35Config::from_value(&serde_json::from_str(config).unwrap()).unwrap();
        let mapper = Qwen35HfNameMapper::new(&config);
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for tensor in mapper
            .tensors()
            .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
        {
            let end = offset + tensor.bytes();
            header.insert(
                tensor.external_name.clone(),
                serde_json::json!({
                    "dtype": tensor.dtype.as_str(), "shape": tensor.shape,
                    "data_offsets": [offset, end]
                }),
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + offset).unwrap();
        Self { dir }
    }
}

// Child-local environment avoids unsafe process-global mutation in parallel tests.
pub fn with_workspace(test_name: &str, body: impl FnOnce()) {
    if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() == Ok(":0:0") {
        body();
        return;
    }
    let test_name = test_name.split_once("::").unwrap().1;
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env("CUBLAS_WORKSPACE_CONFIG", ":0:0")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}
