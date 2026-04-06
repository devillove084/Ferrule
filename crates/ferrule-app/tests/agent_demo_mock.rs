mod common;

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

#[test]
fn agent_demo_mock_succeeds() {
    let fixture = common::make_mock_fixture();
    let agent_cfg = fixture.model_dir.parent().unwrap().join("agent.toml");

    fs::write(
        &agent_cfg,
        format!(
            r#"[observability]
service_name = "ferrule-test"
log_level = "info"
log_format = "json"
metrics_enabled = false
metrics_bind = "127.0.0.1:0"

[model]
backend = "mock"
model_id = "{}"
family = "llama"
device = "cpu"
chat_template = "plain"
dtype = "f32"
use_flash_attn = false
use_kv_cache = true

[rollout]
max_steps = 8
seed = 42

[agent]
initial_observation = "Please compute 2+2 using the tool."
max_steps = 4
max_new_tokens = 16
temperature = 0.2
top_p = 0.9
top_k = 20
expected_final = "4"
"#,
            fixture.model_dir.display()
        ),
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("ferrule-app").unwrap();
    cmd.arg("agent-demo")
        .arg("--config")
        .arg(&agent_cfg)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"total_reward\": 1.0"))
        .stdout(predicate::str::contains("TOOL_CALL"))
        .stdout(predicate::str::contains("FINAL: 4"));
}
