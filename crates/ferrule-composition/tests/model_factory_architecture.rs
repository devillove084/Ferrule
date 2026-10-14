//! Narrow source-level boundary guard, backed by the synthetic catalog adapter test.
const FACTORY: &str = include_str!("../src/model_factory/mod.rs");
const PIPELINE: &str = include_str!("../src/model_factory/pipeline.rs");
const SELECTION: &str = include_str!("../src/model_factory/selection.rs");

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split_once(start)
        .expect("guard start marker")
        .1
        .split_once(end)
        .expect("guard end marker")
        .0
}

#[test]
fn shared_execution_policy_consumes_catalog_strategy_not_model_family() {
    for source in [
        section(
            FACTORY,
            "impl ServingModelOptions {",
            "/// Frozen metadata selection",
        ),
        section(FACTORY, "impl ResolvedExecution {", "fn build_pipeline("),
        section(
            FACTORY,
            "    pub fn prepare_serving(",
            "    pub fn prepare(",
        ),
        section(
            PIPELINE,
            "    pub fn prepare_pipeline(",
            "impl ResidentModelBuildPlan {",
        ),
        section(
            PIPELINE,
            "    pub fn with_pipeline(",
            "// These adapters only bind checkpoint",
        ),
        SELECTION,
    ] {
        for forbidden in [
            "ModelFamily::",
            "Qwen35",
            "qwen35::",
            "qwen35_composition::",
        ] {
            assert!(
                !source.contains(forbidden),
                "shared policy rediscovered family via {forbidden}"
            );
        }
    }
}

#[test]
fn public_engine_kind_is_derived_from_typed_constructor() {
    let kind = section(FACTORY, "pub enum SelectedImplementationKind {", "}");
    assert_eq!(
        kind.split_whitespace().collect::<Vec<_>>(),
        ["Resident,", "Pipeline,"]
    );
    let execution = section(
        FACTORY,
        "struct ResolvedExecution {",
        "impl ResolvedExecution",
    );
    assert!(execution.contains("construction: ResolvedConstruction"));
    assert!(!execution.contains("kind:"));
    assert!(!execution.contains("build:"));
    assert!(SELECTION.contains("Resident(EngineBuilder)"));
    assert!(SELECTION.contains("Pipeline(EngineBuilder)"));
    let qwen = include_str!("../src/model_factory/adapters/qwen35/mod.rs");
    let build = section(qwen, "pub(super) fn build(request:", "if request.family ==");
    assert!(!build.contains("SelectedImplementationKind"));
    assert!(!build.contains(".kind"));
    assert!(!build.contains("expert_placement"));
}

#[test]
fn complete_requests_and_resolved_facts_cross_the_shared_boundary() {
    for source in [
        section(
            FACTORY,
            "    fn prepare_with_entry(",
            "/// Sendable resident-model construction inputs",
        ),
        section(
            FACTORY,
            "fn configure_selected_request(",
            "fn optional_mebibytes_to_bytes(",
        ),
        section(
            FACTORY,
            "    pub fn effective_options(",
            "    pub const fn model_name(",
        ),
        SELECTION,
    ] {
        for forbidden in [
            "ModelFamily::",
            "qwen35::",
            "qwen35_composition::",
            ".qwen35()",
            "resident_backend_name()",
        ] {
            assert!(
                !source.contains(forbidden),
                "shared lowering/reporting rediscovered provider via {forbidden}"
            );
        }
    }
    let prepare = section(
        FACTORY,
        "    pub fn prepare_serving(",
        "    pub fn prepare(",
    );
    for forbidden in [
        "model_profile",
        "model_precision",
        "kv_execution_policy",
        "prepare_dense",
        "prepare_mixed",
    ] {
        assert!(
            !prepare.contains(forbidden),
            "public selection bypassed complete catalog request via {forbidden}"
        );
    }
    let conversion = section(
        PIPELINE,
        "    pub(super) fn validate_pipeline_conversion(",
        "    pub(super) fn with_resolved_pipeline(",
    );
    assert!(conversion.contains("execution.conversion"));
    for forbidden in [
        "StorageEncoding",
        "ArithmeticPrecision",
        "kv_execution_policy",
    ] {
        assert!(!conversion.contains(forbidden));
    }
    assert!(!PIPELINE.contains("hybrid_tensor"));
    assert!(!PIPELINE.contains("validate_inner"));
    for source in [
        include_str!("../src/model_factory/bf16_serving.rs"),
        include_str!("../src/model_factory/mixed_serving.rs"),
        include_str!("../src/model_factory/dense_f32_kv_policy.rs"),
    ] {
        assert!(!source.contains("prepare_with_entry("));
        assert!(!source.contains("prepare_selected("));
    }
}

#[test]
fn shared_provider_sources_cannot_reach_family_adapters_even_through_aliases() {
    fn visit(path: &std::path::Path) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path);
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            let compact = source.split_whitespace().collect::<String>().to_lowercase();
            for forbidden in [
                "qwen",
                "adapters::",
                "family_options",
                "modelfamily::",
                "model_factory::*",
                "super::super::*",
            ] {
                assert!(
                    !compact.contains(forbidden),
                    "shared provider {} reaches a family dependency via {forbidden}",
                    path.display()
                );
            }
        }
    }
    visit(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model_factory/parallel"));
}

#[test]
fn family_binding_is_a_leaf_and_process_wire_selection_is_unchanged() {
    let adapter = include_str!("../src/model_factory/adapters/qwen35/binding.rs");
    assert!(adapter.contains("Qwen35Adapter::bind_hf_metadata"));
    assert!(adapter.contains("Qwen35Profile::Moe35BA3Bfp8"));
    assert!(adapter.contains("NumericProviderInputs"));
    assert!(adapter.contains("parallel::plan::Plan::new"));
    let child = include_str!("../src/process_binding.rs");
    assert!(child.contains("DecoderRecipeKind::Qwen35DenseHybrid"));
    assert!(child.contains("numeric process requires strict Qwen3.5-35B-A3B FP8"));
}
