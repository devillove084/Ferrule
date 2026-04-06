mod common;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn rollout_mock_produces_rewarded_trajectory() {
    let fixture = common::make_mock_fixture();

    let mut cmd = Command::cargo_bin("ferrule-app").unwrap();
    cmd.arg("rollout")
        .arg("--config")
        .arg(&fixture.config_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"total_reward\": 1.0"))
        .stdout(predicate::str::contains("mock_done"));
}
