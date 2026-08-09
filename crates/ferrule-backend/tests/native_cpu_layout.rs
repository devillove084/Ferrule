use std::collections::BTreeMap;
use std::process::Command;

const BUILD_RS: &str = include_str!("../build.rs");
const CPU_NATIVE_ROOT: &str = "native/cpu";
const FORBIDDEN_DIRECTORIES: &[&str] = &["capabilities", "schedules"];
const NATIVE_SOURCE_EXTENSIONS: &[&str] = &["c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx"];

#[test]
fn native_cpu_tree_uses_coarse_grained_directories() {
    let source_files = native_cpu_source_files();
    assert!(
        !source_files.is_empty(),
        "native CPU provider sources must exist under {CPU_NATIVE_ROOT}"
    );
    let forbidden_paths = source_files
        .iter()
        .filter(|path| {
            path_components(path).any(|component| FORBIDDEN_DIRECTORIES.contains(&component))
        })
        .cloned()
        .collect::<Vec<_>>();

    let mut files_by_directory = BTreeMap::<String, Vec<String>>::new();
    for source_file in &source_files {
        let (directory, _) = source_file
            .rsplit_once('/')
            .unwrap_or_else(|| panic!("native CPU source has no parent: {source_file}"));
        files_by_directory
            .entry(directory.to_owned())
            .or_default()
            .push(source_file.to_owned());
    }
    let singleton_directories = files_by_directory
        .into_iter()
        .filter(|(_, files)| files.len() == 1)
        .map(|(directory, files)| format!("{directory} ({})", files[0]))
        .collect::<Vec<_>>();

    assert!(
        forbidden_paths.is_empty(),
        "native CPU sources must not use capabilities/schedules directories:\n{}",
        forbidden_paths.join("\n")
    );
    assert!(
        singleton_directories.is_empty(),
        "native CPU directories must contain coarse-grained source groups:\n{}",
        singleton_directories.join("\n")
    );
}

#[test]
fn build_uses_the_canonical_native_cpu_provider() {
    assert!(
        BUILD_RS.contains("native/cpu"),
        "build.rs must declare the canonical native CPU source root"
    );
    assert!(
        BUILD_RS.contains("provider.cc"),
        "build.rs must compile the canonical native CPU provider entrypoint"
    );
    for forbidden in ["cpu/capabilities", "cpu/schedules", "cpu/implementations"] {
        assert!(
            !BUILD_RS.contains(forbidden),
            "build.rs must not reference legacy CPU build paths: {forbidden}"
        );
    }
}

fn native_cpu_source_files() -> Vec<String> {
    let mut files = run_rg(&["--files", "--hidden", "--no-ignore", "native"])
        .lines()
        .map(normalize_path)
        .filter(|path| path.starts_with(&format!("{CPU_NATIVE_ROOT}/")))
        .filter(|path| is_native_source(path))
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

fn path_components(path: &str) -> impl Iterator<Item = &str> {
    path.split(['/', '\\'])
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
