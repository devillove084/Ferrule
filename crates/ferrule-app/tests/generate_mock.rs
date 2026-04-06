mod common;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn generate_mock_succeeds() {
    let fixture = common::make_mock_fixture();

    let mut cmd = Command::cargo_bin("ferrule-app").unwrap();
    cmd.arg("generate")
        .arg("--config")
        .arg(&fixture.config_path)
        .arg("--prompt")
        .arg("Hello Ferrule")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"finish_reason\""))
        .stdout(predicate::str::contains("\"token_ids\""));
}
