use super::*;
use ferrule_composition::model_factory::NumericFp8Profile;

#[test]
fn numeric_profile_cli_is_explicit_for_chat_and_serve() {
    for command in ["chat", "serve"] {
        for (flag, expected) in [
            (None, None),
            (Some("f32"), Some(NumericFp8Profile::F32)),
            (Some("w8a16-bf16"), Some(NumericFp8Profile::W8A16Bf16)),
        ] {
            let mut argv = vec!["ferrule", command, "not-opened"];
            if let Some(flag) = flag {
                argv.extend(["--numeric-fp8-precision", flag]);
            }
            let args = match Cli::try_parse_from(argv).unwrap().command {
                Command::Chat { sampling, .. } => sampling.cuda_expert_device,
                Command::Serve(args) => args.cuda_expert_device,
                _ => unreachable!(),
            };
            assert_eq!(args.numeric_profile(), expected);
            assert!(
                args.capacity_limits().is_none(),
                "precision must not become a single-GPU cache override on EP"
            );
        }
        for unknown in ["bf16", "auto", "w8a16", "unknown"] {
            assert!(
                Cli::try_parse_from([
                    "ferrule",
                    command,
                    "not-opened",
                    "--numeric-fp8-precision",
                    unknown
                ])
                .is_err()
            );
        }
    }
}
