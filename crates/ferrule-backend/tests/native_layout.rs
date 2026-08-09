use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const NATIVE_ROOT: &str = "native";
const CUDA_ROOT: &str = "native/cuda";
const CUDA_SLICES: &[&str] = &["core", "cutlass"];
const CUDA_NATIVE_FILES: &[&str] = &[
    "native/cuda/core/abi.h",
    "native/cuda/core/attention_ops.cuh",
    "native/cuda/core/bindings.cuh",
    "native/cuda/core/dense_ops.cuh",
    "native/cuda/core/device.cuh",
    "native/cuda/core/moe_ops.cuh",
    "native/cuda/core/provider.cu",
    "native/cuda/core/sequence_ops.cuh",
    "native/cuda/core/tensor_ops.cuh",
    "native/cuda/core/validation.cuh",
    "native/cuda/cutlass/abi.h",
    "native/cuda/cutlass/abi_checks.cuh",
    "native/cuda/cutlass/attention.cuh",
    "native/cuda/cutlass/attention_oracle.cuh",
    "native/cuda/cutlass/bindings.cuh",
    "native/cuda/cutlass/decoder_ops.cuh",
    "native/cuda/cutlass/fp4.cuh",
    "native/cuda/cutlass/fp8.cuh",
    "native/cuda/cutlass/manifest.cuh",
    "native/cuda/cutlass/moe.cuh",
    "native/cuda/cutlass/projections.cuh",
    "native/cuda/cutlass/provider.cu",
    "native/cuda/cutlass/target.cuh",
];
const FORBIDDEN_NATIVE_DIRECTORIES: &[&str] = &[
    "abi",
    "architectures",
    "capabilities",
    "common",
    "diagnostics",
    "implementations",
    "operators",
    "providers",
    "schedules",
];
const FORMER_CORE_NAME: &str = concat!("port", "able");
const LEGACY_CUDA_SOURCE_PATHS: &[&str] = &[
    "src/cuda/cutlass.rs",
    "src/cuda/kernels.rs",
    "src/cuda/kv_page_pool.rs",
    "src/cuda/provider.rs",
    "src/cuda/transformer/mod.rs",
    "src/cuda/transformer/combined_ring.rs",
    "src/cuda/transformer/compressor_recurrent.rs",
    "src/cuda/transformer/sparse_attention.rs",
];
const LEGACY_CUDA_MODULE_REFERENCE: &str = r"(?:crate|ferrule_backend)[[:space:]]*::[[:space:]]*cuda[[:space:]]*::[[:space:]]*(?:cutlass|kernels|kv_page_pool|provider|transformer)(?:[[:space:]]*::[[:space:]]*[A-Za-z_][A-Za-z0-9_]*)+";
const LEGACY_CUDA_MODULE_DECLARATION: &str = r"pub[[:space:]]+mod[[:space:]]+(?:cutlass|kernels|kv_page_pool|provider|transformer)[[:space:]]*(?:\{|;)";
const NATIVE_SOURCE_EXTENSIONS: &[&str] = &[
    "c", "cc", "cpp", "cxx", "cu", "cuh", "h", "hh", "hpp", "hxx",
];
const PRIVATE_EXTERN_BLOCK_FILES: &[&str] = &[
    "src/cpu/provider.rs",
    "src/cuda/ffi/core.rs",
    "src/cuda/ffi/cutlass.rs",
    "src/cuda/runtime.rs",
];

#[test]
fn native_tree_has_no_empty_or_forbidden_directories() {
    let directories = recursive_directories(NATIVE_ROOT);
    let empty = directories
        .iter()
        .filter(|directory| directory_is_empty(directory))
        .map(|directory| normalize_path(&directory.to_string_lossy()))
        .collect::<Vec<_>>();
    let forbidden = directories
        .iter()
        .filter(|directory| {
            directory.components().any(|component| {
                FORBIDDEN_NATIVE_DIRECTORIES
                    .contains(&component.as_os_str().to_string_lossy().as_ref())
            })
        })
        .map(|directory| normalize_path(&directory.to_string_lossy()))
        .collect::<Vec<_>>();

    assert!(
        empty.is_empty(),
        "native tree contains empty directories:\n{}",
        empty.join("\n")
    );
    assert!(
        forbidden.is_empty(),
        "native tree contains forbidden organizational layers:\n{}",
        forbidden.join("\n")
    );
}

#[test]
fn cuda_native_tree_is_two_flat_vertical_slices() {
    let cuda_root = manifest_path(CUDA_ROOT);
    let mut top_level = fs::read_dir(&cuda_root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", cuda_root.display()))
        .map(|entry| {
            let entry = entry.unwrap_or_else(|error| panic!("failed to read CUDA entry: {error}"));
            assert!(
                entry.path().is_dir(),
                "CUDA root may contain only slice directories: {}",
                entry.path().display()
            );
            entry.file_name().to_string_lossy().into_owned()
        })
        .collect::<Vec<_>>();
    top_level.sort();
    assert_eq!(top_level, CUDA_SLICES);

    let nested = recursive_directories(CUDA_ROOT)
        .into_iter()
        .filter(|directory| {
            directory
                .strip_prefix(&cuda_root)
                .is_ok_and(|relative| relative.components().count() > 1)
        })
        .map(|directory| normalize_path(&directory.to_string_lossy()))
        .collect::<Vec<_>>();
    assert!(
        nested.is_empty(),
        "CUDA slices must stay flat; one-file operator directories are forbidden:\n{}",
        nested.join("\n")
    );
}

#[test]
fn cuda_scope_uses_core_naming_only() {
    let mut forbidden_paths = rg_files(CUDA_ROOT)
        .into_iter()
        .chain(rg_files("src/cuda"))
        .filter(|path| path.to_ascii_lowercase().contains(FORMER_CORE_NAME))
        .collect::<Vec<_>>();
    forbidden_paths.sort();

    let content_matches = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--ignore-case",
        FORMER_CORE_NAME,
        CUDA_ROOT,
        "src/cuda",
    ]);

    assert!(
        forbidden_paths.is_empty(),
        "CUDA paths must use core rather than the former provider name:\n{}",
        forbidden_paths.join("\n")
    );
    assert!(
        content_matches.trim().is_empty(),
        "CUDA namespaces, modules, provider names, comments, and tests must use core:\n{content_matches}"
    );
}

#[test]
fn cuda_aggregate_translation_units_cover_each_file_once() {
    let native_files = native_source_files()
        .into_iter()
        .filter(|path| path.starts_with("native/cuda/"))
        .collect::<BTreeSet<_>>();
    let expected_files = CUDA_NATIVE_FILES
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        native_files, expected_files,
        "CUDA native layout must contain the audited 10 core and 13 CUTLASS files"
    );

    let translation_units = native_files
        .iter()
        .filter(|path| path.ends_with(".cu"))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        translation_units,
        [
            "native/cuda/core/provider.cu".to_owned(),
            "native/cuda/cutlass/provider.cu".to_owned(),
        ],
        "CUDA must have exactly two aggregate provider translation units"
    );

    let mut owners = BTreeMap::<String, Vec<String>>::new();
    for translation_unit in &translation_units {
        let mut closure = BTreeSet::new();
        collect_local_include_closure(translation_unit, &native_files, &mut closure);
        for source in closure {
            owners
                .entry(source)
                .or_default()
                .push(translation_unit.clone());
        }
    }

    let orphaned = native_files
        .iter()
        .filter(|source| !owners.contains_key(*source))
        .cloned()
        .collect::<Vec<_>>();
    let duplicated = owners
        .iter()
        .filter(|(_, source_owners)| source_owners.len() != 1)
        .map(|(source, source_owners)| format!("{source}: {}", source_owners.join(", ")))
        .collect::<Vec<_>>();

    assert!(
        orphaned.is_empty(),
        "CUDA include closure has orphaned sources:\n{}",
        orphaned.join("\n")
    );
    assert!(
        duplicated.is_empty(),
        "CUDA sources belong to multiple aggregate TUs:\n{}",
        duplicated.join("\n")
    );
}

#[test]
fn legacy_cuda_files_and_module_paths_do_not_return() {
    let source_files = rg_files("src");
    let native_files = rg_files(NATIVE_ROOT);
    let legacy_files = source_files
        .iter()
        .chain(&native_files)
        .filter(|path| LEGACY_CUDA_SOURCE_PATHS.contains(&path.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let legacy_declarations = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        LEGACY_CUDA_MODULE_DECLARATION,
        "src/cuda/mod.rs",
    ]);
    let legacy_references = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--glob=*.rs",
        LEGACY_CUDA_MODULE_REFERENCE,
        "src",
    ]);

    assert!(
        legacy_files.is_empty(),
        "legacy CUDA files must not return:\n{}",
        legacy_files.join("\n")
    );
    assert!(
        legacy_declarations.trim().is_empty(),
        "legacy top-level CUDA compatibility modules must not return:\n{legacy_declarations}"
    );
    assert!(
        legacy_references.trim().is_empty(),
        "backend source must use canonical CUDA operator/provider paths:\n{legacy_references}"
    );
}

#[test]
fn ffi_modules_remain_backend_private() {
    let module_declarations = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        r"(?:pub(?:[[:space:]]*\([^)]*\))?[[:space:]]+)?mod[[:space:]]+ffi[[:space:]]*;",
        "src",
    ]);
    let external_references = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--glob=*.rs",
        r"ferrule_backend[[:space:]]*::[[:space:]]*cuda[[:space:]]*::[[:space:]]*ffi",
        "../",
    ]);

    let declarations = module_declarations.lines().collect::<Vec<_>>();
    assert_eq!(
        declarations.len(),
        1,
        "FFI must have exactly one private backend module declaration:\n{module_declarations}"
    );
    assert!(
        declarations[0].starts_with("src/cuda/mod.rs:") && declarations[0].ends_with(":mod ffi;"),
        "the backend FFI module declaration must remain private:\n{module_declarations}"
    );
    assert!(
        external_references.trim().is_empty(),
        "backend FFI must not be referenced outside its private module:\n{external_references}"
    );
}

#[test]
fn native_extern_blocks_remain_in_private_backend_modules() {
    let matches = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        r#"unsafe[[:space:]]+extern[[:space:]]+\"(?:C|system)\"[[:space:]]*\{"#,
        "src",
    ]);
    let unexpected = matches
        .lines()
        .filter(|line| {
            let file = line.split_once(':').map(|(file, _)| file).unwrap_or(line);
            !PRIVATE_EXTERN_BLOCK_FILES.contains(&file)
        })
        .collect::<Vec<_>>();

    assert!(
        unexpected.is_empty(),
        "native extern blocks must remain in private backend CPU provider/CUDA runtime/FFI modules:\n{}",
        unexpected.join("\n")
    );
}

#[test]
fn build_script_does_not_reference_legacy_entrypoints() {
    let matches = run_rg(&[
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--ignore-case",
        "legacy|entrypoints?",
        "build.rs",
    ]);
    assert!(
        matches.trim().is_empty(),
        "build.rs must compile canonical provider sources, not legacy entrypoints:\n{matches}"
    );
}

fn collect_local_include_closure(
    source: &str,
    native_files: &BTreeSet<String>,
    closure: &mut BTreeSet<String>,
) {
    if !closure.insert(source.to_owned()) {
        return;
    }
    let source_path = manifest_path(source);
    let contents = fs::read_to_string(&source_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", source_path.display()));
    for line in contents.lines() {
        let line = line.trim();
        let Some(include) = line
            .strip_prefix("#include \"")
            .and_then(|line| line.strip_suffix('"'))
        else {
            continue;
        };
        let included_source = format!("native/cuda/{include}");
        assert!(
            native_files.contains(&included_source),
            "{source} includes missing local CUDA source {included_source}"
        );
        collect_local_include_closure(&included_source, native_files, closure);
    }
}

fn native_source_files() -> Vec<String> {
    let mut files = run_rg(&["--files", "--hidden", "--no-ignore", NATIVE_ROOT])
        .lines()
        .filter(|line| is_native_source(line))
        .map(normalize_path)
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn is_native_source(path: &str) -> bool {
    path.rsplit_once('.').is_some_and(|(_, extension)| {
        NATIVE_SOURCE_EXTENSIONS.contains(
            &extension
                .trim_end_matches('\r')
                .to_ascii_lowercase()
                .as_str(),
        )
    })
}

fn recursive_directories(root: &str) -> Vec<PathBuf> {
    fn visit(directory: &Path, directories: &mut Vec<PathBuf>) {
        let entries = fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()));
        for entry in entries {
            let path = entry
                .unwrap_or_else(|error| panic!("failed to read directory entry: {error}"))
                .path();
            if path.is_dir() {
                directories.push(path.clone());
                visit(&path, directories);
            }
        }
    }

    let mut directories = Vec::new();
    visit(&manifest_path(root), &mut directories);
    directories
}

fn directory_is_empty(directory: &Path) -> bool {
    fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", directory.display()))
        .next()
        .is_none()
}

fn manifest_path(relative: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
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

fn rg_files(root: &str) -> Vec<String> {
    let mut files = run_rg(&["--files", "--hidden", "--no-ignore", root])
        .lines()
        .filter(|line| !line.is_empty())
        .map(normalize_path)
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn normalize_path(path: &str) -> String {
    path.trim_end_matches('\r').replace('\\', "/")
}
