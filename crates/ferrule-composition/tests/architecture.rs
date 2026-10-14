//! Composition is the application layer, never a runtime/model dependency.

#[test]
fn concrete_factories_have_one_owner_and_no_reverse_dependency() {
    let composition = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for path in [
        "src/model_factory/mod.rs",
        "src/model_factory/parallel/plan.rs",
        "src/model_factory/parallel/budget.rs",
        "src/model_factory/parallel/experts.rs",
        "src/model_factory/parallel/assembly.rs",
        "src/model_factory/resident_f32_bound.rs",
    ] {
        assert!(
            composition.join(path).is_file(),
            "missing canonical owner: {path}"
        );
    }
    for retired in [
        "hybrid_cuda_pp_ep_factory.rs",
        "hybrid_cuda_tp_ep_factory.rs",
    ] {
        assert!(!composition.join("src/model_factory").join(retired).exists());
    }
    let hybrid = include_str!("../src/model_factory/hybrid_pipeline.rs");
    assert!(!hybrid.contains("hybrid_cuda_tp_ep_factory"));
    assert!(!hybrid.contains("hybrid_cuda_pp_ep_factory"));
    assert!(!hybrid.contains(".tensor_parallel"));
    assert!(hybrid.contains("parallel::assembly::build_bound"));
    assert!(
        include_str!("../src/model_factory/pipeline.rs")
            .contains("parallel::assembly::build_bound")
    );
    let runtime = composition.join("../ferrule-runtime");
    assert!(!runtime.join("src/engine/model_factory").exists());
    assert!(!runtime.join("tests/unit/engine/model_factory").exists());
    for manifest in [
        include_str!("../../ferrule-runtime/Cargo.toml"),
        include_str!("../../ferrule-model/Cargo.toml"),
        include_str!("../../ferrule-backend/Cargo.toml"),
        include_str!("../../ferrule-common/Cargo.toml"),
    ] {
        assert!(
            !manifest.contains("ferrule-composition"),
            "lower layers must not depend on composition, even in dev-dependencies"
        );
    }
}

const RUNTIME_FORBIDDEN: &str = r"\b(models|families)\s*(::|[,}]|as\b)|\bferrule_composition\b|\bmodel_factory\b|ModelFamily\s*::\s*(Qwen\w*|Deep[Ss]eek\w*)|\b(Qwen|Deep[Ss]eek)\w*(Adapter|Recipe|Profile|Config|Model)\b|\bferrule_backend\s*::\s*(?:\{\s*(?:[^{};]*,\s*)?)?cuda\s*::\s*(?:\{\s*(?:[^{};]*,\s*)?)?(operator_owner|context|qwen|transformer)\b";

#[test]
fn runtime_sources_have_no_concrete_family_or_factory_backedge() {
    // Extend the runtime's existing parallel guard to every production subtree.
    // ModelFamily::Unknown and generic runner/descriptor contracts remain legal.
    let output = std::process::Command::new("rg")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args([
            "--multiline",
            "--line-number",
            "--glob",
            "*.rs",
            RUNTIME_FORBIDDEN,
            "../ferrule-runtime/src",
        ])
        .output()
        .expect("architecture guard requires ripgrep on PATH");
    assert_eq!(
        output.status.code(),
        Some(1),
        "runtime must not import concrete families or application factories:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn runtime_import_guard_distinguishes_concrete_families_from_generic_contracts() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    for (source, forbidden) in [
        ("use ferrule_model::models::qwen3::Qwen3DenseAdapter;", true),
        ("use ferrule_model::{\n families::{qwen3},\n};", true),
        ("use ferrule_model::{models as builtins};", true),
        ("use ferrule_composition as factories;", true),
        ("use crate::engine::model_factory::build;", true),
        ("ModelFamily::Qwen35", true),
        (
            "use ferrule_model::models::deepseek_v4::DeepSeekV4Adapter;",
            true,
        ),
        (
            "use ferrule_backend::cuda::operator_owner::CudaOperators;",
            true,
        ),
        ("use ferrule_backend::cuda::context::CudaOperators;", true),
        (
            "use ferrule_backend::cuda::{operator_owner::CudaOperators};",
            true,
        ),
        ("use ferrule_backend::cuda::{context::CudaOperators};", true),
        (
            "use ferrule_backend::{cuda::operator_owner::CudaOperators};",
            true,
        ),
        ("use ferrule_backend::{cuda::context::CudaOperators};", true),
        (
            "use ferrule_backend::{\n cuda::{operator_owner as owner},\n};",
            true,
        ),
        (
            "use ferrule_backend::{\n cuda::{context as owner},\n};",
            true,
        ),
        (
            "use ferrule_backend::{resources, cuda::{operators, operator_owner}};",
            true,
        ),
        (
            "use ferrule_backend::{resources, cuda::{operators, context}};",
            true,
        ),
        ("use ferrule_backend::cuda::qwen;", true),
        ("use ferrule_backend::cuda::transformer;", true),
        ("use ferrule_backend::cuda::{qwen as family};", true),
        (
            "use ferrule_backend::{cuda::{transformer as family}};",
            true,
        ),
        (
            "use ferrule_backend::cuda::operators::linear::CudaOperators;",
            false,
        ),
        (
            "use ferrule_backend::{cuda::{operators::linear::CudaOperators}};",
            false,
        ),
        (
            "use ferrule_model::{ModelFamily, decoder::MultiSessionRunner};",
            false,
        ),
        ("ModelFamily::Unknown(\"mock\".into())", false),
    ] {
        let mut child = Command::new("rg")
            .args(["--multiline", RUNTIME_FORBIDDEN, "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .expect("architecture guard requires ripgrep on PATH");
        child
            .stdin
            .take()
            .unwrap()
            .write_all(source.as_bytes())
            .unwrap();
        assert_eq!(
            child.wait().unwrap().code(),
            Some(if forbidden { 0 } else { 1 }),
            "unexpected architecture classification: {source}"
        );
    }
}

#[test]
fn cli_features_forward_server_and_keep_reference_execution_opt_in() {
    fn feature<'a>(manifest: &'a str, name: &str) -> &'a str {
        manifest
            .split_once(&format!("\n{name} = ["))
            .expect("missing feature")
            .1
            .split_once(']')
            .expect("unterminated feature")
            .0
    }

    let cli = include_str!("../../ferrule-cli/Cargo.toml");
    assert!(feature(cli, "default").trim().is_empty());
    assert!(feature(cli, "cuda").contains("\"ferrule-server/cuda\""));
    assert!(!feature(cli, "cuda").contains("cpu-reference"));
    assert!(feature(cli, "cpu-reference").contains("\"ferrule-server/cpu-reference\""));
    assert!(!feature(cli, "cpu-reference").contains("/cuda"));

    let server = include_str!("../../ferrule-server/Cargo.toml");
    assert!(feature(server, "cuda").contains("\"ferrule-composition/cuda\""));
    assert!(feature(server, "cpu-reference").contains("\"ferrule-composition/cpu-reference\""));
    assert!(
        server
            .split_once("[dev-dependencies]")
            .unwrap()
            .1
            .contains("ferrule-composition"),
        "server composition remains a dev dependency"
    );
}
