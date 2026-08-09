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
const QWEN_PRODUCTION_LOC_BUDGET: usize = 1_000;

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
const RESTRICTED_BACKEND_REFERENCE: &str = r"\bferrule_backend[[:space:]]*::[[:space:]]*(?:cuda[[:space:]]*::[[:space:]]*(?:cutlass|ffi|kernels|provider|providers|qwen|transformer)\b|(?:ffi|native|provider|providers)\b|cpu[[:space:]]*::[[:space:]]*(?:NativeCpuProvider|ReferenceCpuProvider)\b)(?:[[:space:]]*::[[:space:]]*[A-Za-z_][A-Za-z0-9_]*)*";

// Provider-specific backend references have no symbol or occurrence exceptions.
// Every model family is enforced by default.
const DEEPSEEK_BACKEND_PROVIDER_ALLOWLIST: &[&str] = &[];
const QWEN_BACKEND_PROVIDER_ALLOWLIST: &[&str] = &[];

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

    let matches = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--only-matching",
        "--glob=*.rs",
        RESTRICTED_BACKEND_REFERENCE,
        DEEPSEEK_SOURCE_ROOT,
        QWEN_SOURCE_ROOT,
    ]);
    let observed = parse_restricted_references(&matches);
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
    let source_files = rg_files(DEEPSEEK_SOURCE_ROOT, "*.rs");
    let relative_files = source_files
        .iter()
        .map(|file| {
            file.strip_prefix(&format!("{DEEPSEEK_SOURCE_ROOT}/"))
                .unwrap_or_else(|| panic!("rg returned DeepSeek source outside its root: {file}"))
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let final_files = FINAL_DEEPSEEK_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let forbidden_files = FORBIDDEN_DEEPSEEK_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

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

    let test_files = DEEPSEEK_TEST_FILES.iter().copied().collect::<BTreeSet<_>>();
    let production_file_count = relative_files
        .iter()
        .filter(|file| !test_files.contains(file.as_str()))
        .count();

    let mut debt = Vec::new();
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
    if production_file_count > DEEPSEEK_PRODUCTION_FILE_BUDGET {
        debt.push(format!(
            "production file count is {production_file_count}, budget is {DEEPSEEK_PRODUCTION_FILE_BUDGET}"
        ));
    }

    enforce_boundary("DeepSeek model boundary", &debt);
}

#[test]
fn qwen_source_converges_on_the_final_boundary() {
    let source_files = rg_files(QWEN_SOURCE_ROOT, "*.rs");
    let relative_files = source_files
        .iter()
        .map(|file| {
            file.strip_prefix(&format!("{QWEN_SOURCE_ROOT}/"))
                .unwrap_or_else(|| panic!("rg returned Qwen source outside its root: {file}"))
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let final_files = FINAL_QWEN_FILES.iter().copied().collect::<BTreeSet<_>>();
    let forbidden_files = FORBIDDEN_QWEN_FILES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();

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

    let test_files = QWEN_TEST_FILES.iter().copied().collect::<BTreeSet<_>>();
    let production_file_count = relative_files
        .iter()
        .filter(|file| !test_files.contains(file.as_str()))
        .count();
    let mut production_loc = 0;
    let mut production_breakdown = Vec::new();
    for (source_file, relative_file) in source_files.iter().zip(relative_files.iter()) {
        if test_files.contains(relative_file.as_str()) {
            continue;
        }
        let loc = physical_line_count(source_file);
        production_loc += loc;
        production_breakdown.push(format!("{relative_file}: {loc}"));
    }

    let mut debt = Vec::new();
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
    if production_file_count > QWEN_PRODUCTION_FILE_BUDGET {
        debt.push(format!(
            "production file count is {production_file_count}, budget is {QWEN_PRODUCTION_FILE_BUDGET}"
        ));
    }
    if production_loc > QWEN_PRODUCTION_LOC_BUDGET {
        debt.push(format!(
            "production LOC is {production_loc}, budget is {QWEN_PRODUCTION_LOC_BUDGET} ({})",
            production_breakdown.join(", ")
        ));
    }

    enforce_boundary("Qwen model boundary", &debt);
}

fn parse_restricted_references(output: &str) -> BTreeMap<(String, String), Vec<usize>> {
    let mut references = BTreeMap::<(String, String), Vec<usize>>::new();
    for record in output.lines().filter(|line| !line.is_empty()) {
        let (file, rest) = record
            .split_once(':')
            .unwrap_or_else(|| panic!("unexpected rg match record: {record}"));
        let (line, symbol) = rest
            .split_once(':')
            .unwrap_or_else(|| panic!("unexpected rg match record: {record}"));
        let line = line
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("invalid rg line number in {record:?}: {error}"));
        let symbol = symbol
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        references
            .entry((normalize_path(file), symbol))
            .or_default()
            .push(line);
    }
    references
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
        .unwrap_or_else(|error| panic!("failed to execute rg with {args:?}: {error}"));

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
