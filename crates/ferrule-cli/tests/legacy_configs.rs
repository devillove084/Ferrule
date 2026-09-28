use std::process::Command;

fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .args(args)
        .output()
        .expect("run CLI parser without starting a model")
}

#[test]
fn legacy_examples_are_not_advertised_as_commands_or_runtime_inputs() {
    for args in [&["--help"][..], &["serve", "--help"][..]] {
        let output = cli(args);
        assert!(output.status.success());
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(!help.contains("--config"), "{help}");
        for legacy in ["agent", "train", "rollout"] {
            assert!(!help.contains(&format!("  {legacy} ")), "{help}");
            assert!(!help.contains(&format!("configs/{legacy}.toml")), "{help}");
        }
    }
}

#[test]
fn parser_rejects_legacy_commands_and_toml_flags_before_execution() {
    for command in ["agent", "train", "rollout"] {
        let path = format!("configs/{command}.toml");
        let output = cli(&[command, "--config", &path]);
        assert_eq!(output.status.code(), Some(2));
        let diagnostic = String::from_utf8(output.stderr).unwrap();
        assert!(
            diagnostic.contains("unrecognized subcommand"),
            "{diagnostic}"
        );
        assert!(diagnostic.contains(command), "{diagnostic}");
        for args in [vec!["--config", &path], vec!["serve", "--config", &path]] {
            let output = cli(&args);
            assert_eq!(output.status.code(), Some(2));
            let diagnostic = String::from_utf8(output.stderr).unwrap();
            assert!(
                diagnostic.contains("unexpected argument '--config'"),
                "{diagnostic}"
            );
        }
    }
}
