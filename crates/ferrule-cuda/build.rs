use std::env;
use std::path::{Path, PathBuf};

const CUTLASS_FEATURE_ENV: &str = "CARGO_FEATURE_CUTLASS_SM121A";
const CUTLASS_DIR_ENV: &str = "FERRULE_CUTLASS_DIR";

fn main() {
    println!("cargo:rerun-if-env-changed={CUTLASS_DIR_ENV}");
    println!("cargo:rerun-if-changed=native/cutlass");

    if env::var_os(CUTLASS_FEATURE_ENV).is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo must set CARGO_MANIFEST_DIR"),
    );
    let cutlass_dir = env::var_os(CUTLASS_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../target/vendor/cutlass"));
    require_cutlass_headers(&cutlass_dir);

    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .cpp(true)
        .file(manifest_dir.join("native/cutlass/bridge.cu"))
        .include(manifest_dir.join("native/cutlass"))
        .include(cutlass_dir.join("include"))
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("--expt-extended-lambda")
        .flag("--generate-code=arch=compute_121a,code=sm_121a")
        .compile("ferrule_cutlass_sm121a");
}

fn require_cutlass_headers(cutlass_dir: &Path) {
    let version_header = cutlass_dir.join("include/cutlass/version.h");
    if !version_header.is_file() {
        panic!(
            "CUTLASS headers were not found at {}. Clone NVIDIA CUTLASS v4.6.1 there, or set {CUTLASS_DIR_ENV} to a pinned checkout",
            cutlass_dir.display()
        );
    }

    println!("cargo:rerun-if-changed={}", version_header.display());
}
