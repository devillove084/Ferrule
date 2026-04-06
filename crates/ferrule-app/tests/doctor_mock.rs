mod common;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn doctor_mock_succeeds() {
    let fixture = common::make_mock_fixture();

    let mut cmd = Command::cargo_bin("ferrule-app").unwrap();
    cmd.arg("doctor")
        .arg("--config")
        .arg(&fixture.config_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("\"backend\": \"mock\""))
        .stdout(predicate::str::contains("\"weight_files\": 1"));
}
