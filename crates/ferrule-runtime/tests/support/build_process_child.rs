//! Build test children from this checkout; never discover executables by ancestry.
use std::path::PathBuf;
use std::process::Command;

pub fn build(package: &str, kind: &str, name: &str) -> PathBuf {
    assert!(matches!(kind, "example" | "bin"));
    let mut command = Command::new(env!("CARGO"));
    command
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "build",
            "--locked",
            "--message-format=json",
            "--no-default-features",
            "-p",
            package,
        ])
        .arg(format!("--{kind}"))
        .arg(name);
    if cfg!(feature = "cuda") {
        command.args(["--features", "cuda"]);
    }
    let output = command.output().expect("build matching process test child");
    assert!(
        output.status.success(),
        "process child build failed: {command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );
    // Cargo reports the selected artifact even for a fresh build. This respects
    // target-dir/config overrides without trusting an unrelated existing binary.
    let mut artifacts = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .filter(|message| {
            message["reason"] == "compiler-artifact"
                && message["target"]["name"] == name
                && message["target"]["kind"] == serde_json::json!([kind])
        })
        .filter_map(|message| message["executable"].as_str().map(PathBuf::from))
        .collect::<Vec<_>>();
    assert_eq!(artifacts.len(), 1, "expected one {package} {kind} artifact");
    let executable = artifacts.pop().unwrap();
    assert!(
        executable.is_file(),
        "missing child {}",
        executable.display()
    );
    eprintln!(
        "process test child: {} (cuda={})",
        executable.display(),
        cfg!(feature = "cuda")
    );
    executable
}
