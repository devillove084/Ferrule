use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use std::fs;

#[test]
#[ignore = "requires FERRULE_E2E_MODEL_DIR pointing to a local llama-family model"]
fn generate_real_llama_smoke() {
    let model_dir = env::var("FERRULE_E2E_MODEL_DIR").unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let config_path = tmp.path().join("real-generate.toml");

    fs::write(
        &config_path,
        format!(
            r#"[observability]
service_name = "ferrule-e2e"
log_level = "info"
log_format = "json"
metrics_enabled = false
metrics_bind = "127.0.0.1:0"

[model]
backend = "real"
model_id = "{}"
family = "llama"
device = "auto"
chat_template = "plain"
dtype = "f32"
use_flash_attn = false
use_kv_cache = true

[rollout]
max_steps = 16
seed = 42
"#,
            model_dir
        ),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("ferrule-app").unwrap();
    cmd.arg("generate")
        .arg("--config")
        .arg(&config_path)
        .arg("--prompt")
        .arg("Hello Ferrule")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"text\""))
        .stdout(predicate::str::contains("\"finish_reason\""));
}
