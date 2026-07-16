use std::env;
use std::path::{Path, PathBuf};

mod architecture_target;

use architecture_target::{CudaArchitectureFamily, CudaTarget};

const CUTLASS_FEATURE_ENV: &str = "CARGO_FEATURE_CUTLASS";
const CUTLASS_DIR_ENV: &str = "FERRULE_CUTLASS_DIR";
const CUTLASS_ARCH_ENV: &str = "FERRULE_CUTLASS_ARCH";

fn main() {
    println!("cargo:rerun-if-env-changed={CUTLASS_DIR_ENV}");
    println!("cargo:rerun-if-env-changed={CUTLASS_ARCH_ENV}");
    println!("cargo:rerun-if-env-changed=CUDA_OXIDE_TARGET");
    println!("cargo:rerun-if-changed=native/cutlass");
    println!("cargo:rerun-if-changed=architecture_target.rs");

    for cfg in [
        "ferrule_cuda_hopper_wgmma",
        "ferrule_cuda_blackwell_tcgen05",
        "ferrule_cuda_blackwell_mma_sync_fp8",
        "ferrule_cuda_blackwell_mma_sync_mxfp4",
        "ferrule_cuda_cuda_oxide_bf16_mma",
    ] {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }

    let cuda_target = env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "portable".into());
    println!("cargo:rustc-env=FERRULE_CUDA_COMPILED_TARGET={cuda_target}");
    if let Some(target) = CudaTarget::parse(&cuda_target) {
        let capabilities = target.capabilities();
        if capabilities.hopper_wgmma {
            println!("cargo:rustc-cfg=ferrule_cuda_hopper_wgmma");
        }
        if capabilities.blackwell_tcgen05 {
            println!("cargo:rustc-cfg=ferrule_cuda_blackwell_tcgen05");
        }
        if capabilities.blackwell_mma_sync_fp8 {
            println!("cargo:rustc-cfg=ferrule_cuda_blackwell_mma_sync_fp8");
        }
        if capabilities.blackwell_mma_sync_mxfp4 {
            println!("cargo:rustc-cfg=ferrule_cuda_blackwell_mma_sync_mxfp4");
        }
        if matches!(
            target.family,
            CudaArchitectureFamily::BlackwellDatacenter | CudaArchitectureFamily::BlackwellConsumer
        ) {
            println!("cargo:rustc-cfg=ferrule_cuda_cuda_oxide_bf16_mma");
        }
    }

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

    let cutlass_target = env::var(CUTLASS_ARCH_ENV)
        .ok()
        .or_else(|| env::var("CUDA_OXIDE_TARGET").ok())
        .unwrap_or_else(|| {
            panic!(
                "CUTLASS is a GB10-only provider; set {CUTLASS_ARCH_ENV}=sm_121a or build with --arch sm_121a"
            )
        });
    let parsed_target = CudaTarget::parse(&cutlass_target)
        .unwrap_or_else(|| panic!("invalid CUTLASS target '{cutlass_target}'"));
    if !cutlass_target.starts_with("sm_")
        || parsed_target.compute_capability() != 121
        || !parsed_target.has_accelerated_target()
    {
        panic!("CUTLASS is a GB10-only provider and requires sm_121a, got '{cutlass_target}'");
    }
    let compute_target = cutlass_target.replacen("sm_", "compute_", 1);
    let generate_code = format!("--generate-code=arch={compute_target},code={cutlass_target}");
    let target_sm = parsed_target.compute_capability().to_string();

    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        // CUTLASS/CuTe headers intentionally contain host-side placeholder
        // parameters for device-only paths. Keep Ferrule's Rust warnings on,
        // but do not surface third-party header noise from this translation unit.
        .warnings(false)
        .cpp(true)
        .file(manifest_dir.join("native/cutlass/bridge.cu"))
        .include(manifest_dir.join("native/cutlass"))
        .include(cutlass_dir.join("include"))
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("--expt-extended-lambda")
        .flag(&generate_code)
        .define("FERRULE_CUTLASS_TARGET_SM", Some(target_sm.as_str()))
        .compile("ferrule_cutlass");
}

fn require_cutlass_headers(cutlass_dir: &Path) {
    let version_header = cutlass_dir.join("include/cutlass/version.h");
    if !version_header.is_file() {
        panic!(
            "CUTLASS headers were not found at {}. Run `just cutlass-setup` to fetch v4.6.1 at e05f953a5b3d38adc240df2ff928e0421c2abba3, or set {CUTLASS_DIR_ENV} to that pinned checkout",
            cutlass_dir.display()
        );
    }

    println!("cargo:rerun-if-changed={}", version_header.display());
}
