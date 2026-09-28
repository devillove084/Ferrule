//! Static text guard, not a semantic dependency proof. Requires ripgrep (`rg`)
//! on PATH with --json/--multiline support; no PCRE2 or AST dependency.
use std::collections::{BTreeMap, BTreeSet};

use std::process::Command;

const DEEPSEEK_SOURCE_ROOT: &str = "src/models/deepseek_v4";
const DEEPSEEK_PRODUCTION_FILE_BUDGET: usize = 6;

const FINAL_DEEPSEEK_FILES: &[&str] = &[
    "mod.rs",
    "adapter.rs",
    "checkpoint.rs",
    "config.rs",
    "recipe.rs",
    "name_mapper.rs",
    "tests.rs",
];
const DEEPSEEK_TEST_FILES: &[&str] = &["tests.rs"];
const FORBIDDEN_DEEPSEEK_FILES: &[&str] = &[
    "runner.rs",
    "prepared.rs",
    "checkpoint_binding.rs",
    "attention.rs",
    "mla.rs",
    "helpers.rs",
    "sequence.rs",
    "proposal_attachment.rs",
    "cuda_cache.rs",
    "operators.rs",
    "layer.rs",
];

const QWEN_SOURCE_ROOT: &str = "src/models/qwen3";
const QWEN_PRODUCTION_FILE_BUDGET: usize = 6;
// Dense Qwen3 adds ~300 lines of strict configuration and CPU artifact/runner
// composition. Dense/MoE share graph and schema builders; no family forward is
// added. Keep the six-file allowlist and provider restrictions unchanged.
const QWEN_PRODUCTION_LOC_BUDGET: usize = 1_300;

const FINAL_QWEN_FILES: &[&str] = &[
    "adapter.rs",
    "checkpoint.rs",
    "config.rs",
    "mod.rs",
    "name_mapper.rs",
    "recipe.rs",
    "tests.rs",
];
const QWEN_TEST_FILES: &[&str] = &["tests.rs"];
const FORBIDDEN_QWEN_FILES: &[&str] = &[
    "attention.rs",
    "moe.rs",
    "layer.rs",
    "operators.rs",
    "prepared.rs",
    "runner.rs",
    "sequence.rs",
    "checkpoint_binding.rs",
];

// Match implementation-specific backend modules and public types whose names
// expose the DSV4 model family. The model-neutral cuda::operators facade is
// deliberately allowed. Whitespace is accepted around Rust path separators so
// formatting cannot bypass the guard.
const RESTRICTED_BACKEND_REFERENCE: &str = r"\bferrule_backend[[:space:]]*::[[:space:]]*(?:\{[[:space:]]*)?(?:cuda[[:space:]]*::[[:space:]]*(?:\{[[:space:]]*)?(?:context|runtime|cutlass|ffi|kernels|provider|providers|qwen|transformer)\b|(?:ffi|native|provider|providers)\b|cpu[[:space:]]*::[[:space:]]*(?:\{[[:space:]]*)?(?:provider|NativeCpuProvider|ReferenceCpuProvider)\b)(?:[[:space:]]*::[[:space:]]*[A-Za-z_][A-Za-z0-9_]*)*";

// Provider-specific backend references have no symbol or occurrence exceptions.
// Every model family is enforced by default.
const DEEPSEEK_BACKEND_PROVIDER_ALLOWLIST: &[&str] = &[];
const QWEN_BACKEND_PROVIDER_ALLOWLIST: &[&str] = &[];

const QWEN35_SOURCE_ROOT: &str = "src/models/qwen35";
const QWEN35_PRODUCTION_FILE_BUDGET: usize = 6;
// Separate Qwen3.5 schema/config baseline, not an increase to Qwen3's budget.
const QWEN35_PRODUCTION_LOC_BUDGET: usize = 1_900;
const FINAL_QWEN35_FILES: &[&str] = &[
    "adapter.rs",
    "config.rs",
    "metadata.rs",
    "mod.rs",
    "name_mapper.rs",
    "recipe.rs",
    "tests.rs",
];
const QWEN35_TEST_FILES: &[&str] = &["tests.rs"];
const FORBIDDEN_QWEN35_FILES: &[&str] = &[
    "attention.rs",
    "moe.rs",
    "layer.rs",
    "operators.rs",
    "prepared.rs",
    "runner.rs",
    "sequence.rs",
    "checkpoint_binding.rs",
];
const QWEN35_BACKEND_PROVIDER_ALLOWLIST: &[&str] = &[];

/// Every concrete model implementation root is listed here. `common` is an
/// explicitly non-family shared module; a new sibling root must update this
/// manifest before it can evade a boundary check.
const DOCUMENTED_FAMILY_ROOTS: &[&str] = &["deepseek_v4", "qwen3", "qwen35"];
const NON_FAMILY_MODEL_ROOTS: &[&str] = &["common"];
const DOCUMENTED_FAMILY_MANIFEST: &str =
    include_str!("fixtures/architecture_boundaries/family_manifest.txt");

#[test]
fn model_family_roots_are_explicitly_enumerated() {
    let files = rg_files("src/models", "*.rs");
    enforce_boundary(
        "model family discovery",
        &family_discovery_debt(&files, DOCUMENTED_FAMILY_ROOTS),
    );
    let manifest = DOCUMENTED_FAMILY_MANIFEST
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<BTreeSet<_>>();
    assert_eq!(manifest, DOCUMENTED_FAMILY_ROOTS.iter().copied().collect());
    // Listing a new family without also defining/enforcing its file policy is
    // not sufficient. Every documented policy is invoked here.
    for root in DOCUMENTED_FAMILY_ROOTS {
        match *root {
            "deepseek_v4" => deepseek_source_matches_the_final_file_boundary(),
            "qwen3" => qwen_source_converges_on_the_final_boundary(),
            "qwen35" => qwen35_source_converges_on_the_final_boundary(),
            other => panic!("family {other} requires an enforced file/LOC policy"),
        }
    }

    // Tensor-policy family modules are a second family-specific entry surface.
    let semantic_manifest = include_str!("fixtures/architecture_boundaries/semantic_manifest.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<BTreeSet<_>>();
    let semantic_files = rg_files("src/families", "*.rs");
    assert_eq!(
        semantic_files
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        semantic_manifest,
        "unregistered or missing src/families module; document it without exempting provider references"
    );
}

#[test]
fn cuda_transformer_runtime_stays_family_neutral() {
    let cuda = std::fs::read_to_string("src/transformer/cuda.rs")
        .expect("read generic CUDA transformer source");

    assert!(
        !std::path::Path::new("src/models/deepseek_v4/resources.rs").exists(),
        "the DeepSeek resources god file must not return"
    );
    for definition in [
        "pub(crate) struct CudaTransformerRuntime {",
        "pub(crate) type CudaTransformerRuntimeHandle =",
        "pub struct PreparedCudaTransformer {",
    ] {
        assert!(
            cuda.contains(definition),
            "generic CUDA transformer source must own {definition}"
        );
    }
    for forbidden in [
        "PreparedCudaMlaHyperMoeLayer",
        "models::deepseek_v4",
        "DeepSeek",
        "Dsv4",
        "DSV4",
    ] {
        assert!(
            !cuda.contains(forbidden),
            "generic CUDA transformer source contains model-specific residue {forbidden}"
        );
    }
    assert!(
        !cuda.to_ascii_lowercase().contains("deepseek"),
        "generic CUDA transformer source contains a model-family name"
    );
    assert!(
        cuda.contains("PreparedTransformer<"),
        "the CUDA prepared state must aggregate generic prepared transformer components"
    );
}

#[test]
fn removed_transformer_module_stays_absent() {
    assert!(
        !std::path::Path::new("src/transformer/module.rs").exists(),
        "the undeclared transformer/module.rs residue must not return"
    );
}

#[test]
fn deepseek_cuda_only_modules_have_no_cpu_fallbacks() {
    let matches = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--fixed-strings",
        "cfg(not(feature = \"cuda\"))",
        "src/transformer/cuda.rs",
        "src/transformer/attention/mla.rs",
    ]);
    assert!(
        matches.is_empty(),
        "CUDA-only DeepSeek modules must not retain unreachable CPU fallbacks:\n{matches}"
    );
}

#[test]
fn deepseek_uses_generic_paged_kv_custody() {
    assert!(
        !std::path::Path::new("src/models/deepseek_v4/kv.rs").exists(),
        "the deleted DeepSeek KV adapter must not return"
    );
    let matches = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--regexp=DeepSeekKvBackend|DeepSeekKvTransaction|DeepSeekKvStorage|PagedKvOwnership",
        DEEPSEEK_SOURCE_ROOT,
    ]);
    assert!(
        matches.is_empty(),
        "DeepSeek must compose the generic paged KV backend without a model-local custody adapter:
{matches}"
    );
}

#[test]
fn model_does_not_acquire_backend_implementation_types() {
    assert!(
        DEEPSEEK_BACKEND_PROVIDER_ALLOWLIST.is_empty(),
        "DeepSeek backend provider allowlist must remain empty"
    );
    assert!(
        QWEN_BACKEND_PROVIDER_ALLOWLIST.is_empty(),
        "Qwen backend provider allowlist must remain empty"
    );
    assert!(
        QWEN35_BACKEND_PROVIDER_ALLOWLIST.is_empty(),
        "Qwen3.5 backend provider allowlist must remain empty"
    );

    // Scan all files, not only allowed roots/files. Hidden/ignored files and
    // newly added families do not gain a provider-reference exemption.
    let observed = restricted_references("src/models")
        .into_iter()
        .chain(restricted_references("src/families"))
        .collect::<BTreeMap<_, _>>();
    let violations = observed
        .iter()
        .map(|((file, symbol), lines)| {
            format!("{file}: restricted backend reference {symbol} at lines {lines:?}")
        })
        .collect::<Vec<_>>();

    assert!(
        violations.is_empty(),
        "model families must depend on model-neutral execution contracts; remove these references:\n{}",
        violations.join("\n")
    );
}

#[test]
fn deepseek_source_matches_the_final_file_boundary() {
    check_family(
        DEEPSEEK_SOURCE_ROOT,
        FINAL_DEEPSEEK_FILES,
        DEEPSEEK_TEST_FILES,
        FORBIDDEN_DEEPSEEK_FILES,
        DEEPSEEK_PRODUCTION_FILE_BUDGET,
        None,
    );
}

#[test]
fn qwen_source_converges_on_the_final_boundary() {
    check_family(
        QWEN_SOURCE_ROOT,
        FINAL_QWEN_FILES,
        QWEN_TEST_FILES,
        FORBIDDEN_QWEN_FILES,
        QWEN_PRODUCTION_FILE_BUDGET,
        Some(QWEN_PRODUCTION_LOC_BUDGET),
    );
}

#[test]
fn qwen35_source_converges_on_the_final_boundary() {
    check_family(
        QWEN35_SOURCE_ROOT,
        FINAL_QWEN35_FILES,
        QWEN35_TEST_FILES,
        FORBIDDEN_QWEN35_FILES,
        QWEN35_PRODUCTION_FILE_BUDGET,
        Some(QWEN35_PRODUCTION_LOC_BUDGET),
    );
}

fn check_family(
    root: &str,
    allowed: &[&str],
    tests: &[&str],
    forbidden: &[&str],
    budget: usize,
    loc: Option<usize>,
) {
    let files = rg_files(root, "*.rs");
    let relative = files
        .iter()
        .map(|file| file.strip_prefix(&format!("{root}/")).unwrap().to_owned())
        .collect();
    let breakdown = files
        .iter()
        .filter_map(|file| {
            let relative = file.strip_prefix(&format!("{root}/")).unwrap();
            (!tests.contains(&relative)).then(|| (relative.to_owned(), physical_line_count(file)))
        })
        .collect::<Vec<_>>();
    enforce_boundary(
        root,
        &boundary_debt(
            &relative,
            allowed,
            tests,
            forbidden,
            budget,
            loc.map(|limit| (limit, breakdown.as_slice())),
        ),
    );
}

fn restricted_references(root: &str) -> BTreeMap<(String, String), Vec<usize>> {
    // JSON preserves file paths with spaces/colons and multiline matches.
    let output = run_rg(&[
        "--json",
        "--multiline",
        "--hidden",
        "--no-ignore",
        "--glob=*.rs",
        RESTRICTED_BACKEND_REFERENCE,
        root,
    ]);
    let mut references = BTreeMap::<(String, String), Vec<usize>>::new();
    for record in output.lines() {
        let record: serde_json::Value = serde_json::from_str(record).expect("rg JSON record");
        if record["type"] != "match" {
            continue;
        }
        let data = &record["data"];
        let file = data["path"]["text"].as_str().expect("UTF-8 source path");
        let lines = data["lines"]["text"].as_str().unwrap();
        for matched in data["submatches"].as_array().unwrap() {
            let start = matched["start"].as_u64().unwrap() as usize;
            let line = data["line_number"].as_u64().unwrap() as usize
                + lines[..start].bytes().filter(|&b| b == b'\n').count();
            let symbol = matched["match"]["text"]
                .as_str()
                .unwrap()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            references
                .entry((normalize_path(file), symbol))
                .or_default()
                .push(line);
        }
    }
    references
}

#[test]
fn rg_boundary_fixtures_cover_legal_invalid_whitespace_and_new_files() {
    let root = "tests/fixtures/architecture_boundaries";
    let observed = restricted_references(root);
    for (name, line) in [
        ("provider_reference.rs", 1),
        ("grouped_reference.rs", 1),
        ("whitespace reference.rs", 1),
        ("whiteout_new_file.rs", 1),
        ("new_family/adapter.rs", 1),
    ] {
        assert!(
            observed
                .iter()
                .any(|((file, _), lines)| file.ends_with(name) && lines.contains(&line)),
            "missing {name}:{line}: {observed:?}"
        );
    }
    assert!(
        !observed
            .keys()
            .any(|(file, _)| file.ends_with("legal_reference.rs"))
    );
    let legal = BTreeSet::from(["legal_reference.rs".to_owned()]);
    assert!(boundary_debt(&legal, &["legal_reference.rs"], &[], &[], 1, None).is_empty());
    let mut outside = legal;
    outside.insert("whiteout_new_file.rs".into());
    let debt = boundary_debt(&outside, &["legal_reference.rs"], &[], &[], 1, None);
    assert!(
        debt.iter()
            .any(|d| d.contains("outside the final allowlist: whiteout_new_file.rs"))
    );
    assert!(
        debt.iter()
            .any(|d| d.contains("production file count is 2"))
    );
    assert!(
        boundary_debt(&BTreeSet::new(), &["legal_reference.rs"], &[], &[], 1, None)
            .iter()
            .any(|d| d.contains("missing"))
    );
    assert!(
        boundary_debt(
            &outside,
            &["legal_reference.rs", "whiteout_new_file.rs"],
            &[],
            &["whiteout_new_file.rs"],
            2,
            None
        )
        .iter()
        .any(|d| d.contains("forbidden"))
    );
}

#[test]
fn family_discovery_fixture_detects_new_directory_and_flat_roots() {
    let mut files = DOCUMENTED_FAMILY_ROOTS
        .iter()
        .map(|root| format!("src/models/{root}/adapter.rs"))
        .collect::<Vec<_>>();
    files.push("src/models/common/shape.rs".into());
    files.push("src/models/mod.rs".into());
    assert!(family_discovery_debt(&files, DOCUMENTED_FAMILY_ROOTS).is_empty());
    // No mod.rs required: even an undeclared adapter is discovered.
    let discovered = rg_files("tests/fixtures/architecture_boundaries/new_family", "*.rs");
    for file in discovered {
        files.push(file.replace("tests/fixtures/architecture_boundaries/", "src/models/"));
    }
    assert!(
        family_discovery_debt(&files, DOCUMENTED_FAMILY_ROOTS)
            .iter()
            .any(|d| d.contains("new_family"))
    );
    let expanded = &["deepseek_v4", "qwen3", "qwen35", "new_family"];
    assert!(family_discovery_debt(&files, expanded).is_empty());
    files.push("src/models/flat_family.rs".into());
    assert!(
        family_discovery_debt(&files, expanded)
            .iter()
            .any(|d| d.contains("flat_family"))
    );
    assert!(
        !restricted_references("tests/fixtures/architecture_boundaries/new_family").is_empty(),
        "documenting a new family must not exempt its provider references"
    );
}

fn rg_files(root: &str, glob: &str) -> Vec<String> {
    let mut files = run_rg(&["--files", "--hidden", "--no-ignore", "--glob", glob, root])
        .lines()
        .filter(|line| !line.is_empty())
        .map(normalize_path)
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn physical_line_count(file: &str) -> usize {
    let output = run_rg(&["--count-matches", "^", file]);
    let count = output.trim();
    if count.is_empty() {
        0
    } else {
        count
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("invalid LOC count for {file}: {error}"))
    }
}

fn enforce_boundary(name: &str, debt: &[String]) {
    assert!(
        debt.is_empty(),
        "{name} violations:\n- {}",
        debt.join("\n- ")
    );
}

fn run_rg(args: &[&str]) -> String {
    let output = Command::new("rg")
        .arg("--no-config")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap_or_else(|error| {
            panic!("architecture tests require ripgrep (rg) on PATH; {args:?}: {error}")
        });

    if !matches!(output.status.code(), Some(0 | 1)) {
        panic!(
            "rg {args:?} failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("rg returned non-UTF-8 output for {args:?}: {error}"))
}

fn normalize_path(path: &str) -> String {
    path.trim_end_matches('\r').replace('\\', "/")
}

fn family_discovery_debt(files: &[String], documented: &[&str]) -> Vec<String> {
    let roots = files
        .iter()
        .filter_map(|file| file.strip_prefix("src/models/"))
        .filter(|file| *file != "mod.rs")
        .map(|file| file.split('/').next().unwrap().trim_end_matches(".rs"))
        .filter(|root| !NON_FAMILY_MODEL_ROOTS.contains(root))
        .collect::<BTreeSet<_>>();
    let documented = documented.iter().copied().collect::<BTreeSet<_>>();
    roots
        .symmetric_difference(&documented)
        .map(|root| format!("unregistered or missing family root: src/models/{root}"))
        .collect()
}

fn boundary_debt(
    relative_files: &BTreeSet<String>,
    final_files: &[&str],
    test_files: &[&str],
    forbidden_files: &[&str],
    production_file_budget: usize,
    production_loc: Option<(usize, &[(String, usize)])>,
) -> Vec<String> {
    let final_files = final_files.iter().copied().collect::<BTreeSet<_>>();
    let forbidden_files = forbidden_files.iter().copied().collect::<BTreeSet<_>>();
    let test_files = test_files.iter().copied().collect::<BTreeSet<_>>();
    let mut debt = Vec::new();
    let outside_final_set = relative_files
        .iter()
        .filter(|file| !final_files.contains(file.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let missing_final_files = final_files
        .difference(&relative_files.iter().map(String::as_str).collect())
        .copied()
        .collect::<Vec<_>>();
    let forbidden_present = relative_files
        .iter()
        .filter(|file| forbidden_files.contains(file.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let production_file_count = relative_files
        .iter()
        .filter(|file| !test_files.contains(file.as_str()))
        .count();
    if !outside_final_set.is_empty() {
        debt.push(format!(
            "files outside the final allowlist: {}",
            outside_final_set.join(", ")
        ));
    }
    if !missing_final_files.is_empty() {
        debt.push(format!(
            "files missing from the final set: {}",
            missing_final_files.join(", ")
        ));
    }
    if !forbidden_present.is_empty() {
        debt.push(format!(
            "forbidden migration files still present: {}",
            forbidden_present.join(", ")
        ));
    }
    if production_file_count > production_file_budget {
        debt.push(format!(
            "production file count is {production_file_count}, budget is {production_file_budget}"
        ));
    }
    if let Some((budget, breakdown)) = production_loc {
        let total = breakdown.iter().map(|(_, lines)| lines).sum::<usize>();
        if total > budget {
            debt.push(format!(
                "production LOC is {total}, budget is {budget}; files: {breakdown:?}"
            ));
        }
    }
    debt
}
