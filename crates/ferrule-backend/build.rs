use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

mod architecture_target;

use architecture_target::CudaTarget;

const CUDA_FEATURE_ENV: &str = "CARGO_FEATURE_CUDA";
const CUDA_ARCH_ENV: &str = "FERRULE_CUDA_ARCH";
const CUTLASS_DIR_ENV: &str = "FERRULE_CUTLASS_DIR";
const CUDA_TEST_ORACLE_ENV: &str = "FERRULE_CUDA_TEST_ORACLE";

fn main() {
    println!("cargo:rerun-if-env-changed={CUDA_ARCH_ENV}");
    println!("cargo:rerun-if-env-changed={CUTLASS_DIR_ENV}");
    println!("cargo:rerun-if-env-changed={CUDA_TEST_ORACLE_ENV}");
    println!("cargo:rustc-check-cfg=cfg(ferrule_cuda_test_oracle)");
    println!("cargo:rerun-if-changed=native/cuda");
    println!("cargo:rerun-if-changed=architecture_target.rs");

    let cuda_test_oracle = env::var(CUDA_TEST_ORACLE_ENV).is_ok_and(|value| value == "1");
    if cuda_test_oracle {
        println!("cargo:rustc-cfg=ferrule_cuda_test_oracle");
    }

    if env::var_os(CUDA_FEATURE_ENV).is_none() {
        println!("cargo:rustc-env=FERRULE_BACKEND_CUDA_COMPILED_TARGET=portable");
        return;
    }

    let cuda_arch = resolve_cuda_arch();
    let target = CudaTarget::parse(&cuda_arch)
        .unwrap_or_else(|| panic!("invalid CUDA target `{cuda_arch}`; expected sm_XX or sm_XXX"));
    verify_nvcc_support(&cuda_arch, target);
    println!("cargo:rustc-env=FERRULE_BACKEND_CUDA_COMPILED_TARGET={cuda_arch}");
    publish_cuda_driver_search_path();

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo must set CARGO_MANIFEST_DIR"),
    );
    let cutlass_dir = env::var_os(CUTLASS_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("../../target/vendor/cutlass"));
    require_cutlass_headers(&cutlass_dir);

    let native_codegen_target = native_codegen_target(target);
    let compute_target = native_codegen_target.replacen("sm_", "compute_", 1);
    let generate_code =
        format!("--generate-code=arch={compute_target},code={native_codegen_target}");
    let target_sm = target.compute_capability().to_string();
    let capabilities = target.capabilities();
    let capability_flag = |enabled| if enabled { "1" } else { "0" };
    let native_root = manifest_dir.join("native/cuda");
    let implementations_root = native_root.join("implementations");
    let portable_root = implementations_root.join("portable");
    let cutlass_root = implementations_root.join("cutlass");

    let cutlass_include = cutlass_dir.join("include");
    let cutlass_system_include = format!("--system-include={}", cutlass_include.display());

    let configure_implementation = |source: PathBuf| {
        let mut build = cc::Build::new();
        build
            .cuda(true)
            .cudart("shared")
            .warnings(false)
            .cpp(true)
            .file(source)
            .include(&native_root)
            .include(&cutlass_root)
            .flag(&cutlass_system_include)
            .flag("-std=c++17")
            .flag("--expt-relaxed-constexpr")
            .flag("--expt-extended-lambda")
            .flag("--display-error-number")
            .flag("--Werror=all-warnings")
            .flag("-Xcompiler=-Wall,-Wextra")
            .flag(&generate_code)
            .define("FERRULE_CUDA_TARGET_SM", Some(target_sm.as_str()))
            .define(
                "FERRULE_CUDA_HAS_PORTABLE_SIMT",
                Some(capability_flag(capabilities.portable_simt)),
            )
            .define(
                "FERRULE_CUDA_HAS_BF16_MMA_SYNC",
                Some(capability_flag(capabilities.bf16_mma_sync)),
            )
            .define(
                "FERRULE_CUDA_HAS_FP8_MMA_SYNC",
                Some(capability_flag(capabilities.fp8_mma_sync)),
            )
            .define(
                "FERRULE_CUDA_HAS_SM90_WGMMA",
                Some(capability_flag(capabilities.sm90_wgmma)),
            )
            .define(
                "FERRULE_CUDA_HAS_SM1XX_UMMA",
                Some(capability_flag(capabilities.sm1xx_umma)),
            )
            .define(
                "FERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4",
                Some(capability_flag(capabilities.sm103_block_scaled_fp4)),
            )
            .define(
                "FERRULE_CUDA_HAS_SM12X_MXFP4_MMA_SYNC",
                Some(capability_flag(capabilities.sm12x_mxfp4_mma_sync)),
            );
        if cuda_test_oracle {
            build.define("FERRULE_CUDA_TEST_ORACLE", Some("1"));
        }
        build
    };

    // Keep each implementation in its own CUDA compilation unit. cc-rs enables
    // relocatable device code automatically when one Build contains multiple
    // .cu files; that makes ptxas discard CUTLASS warp-group setmaxnreg
    // directives even though these implementations share no device symbols.
    configure_implementation(portable_root.join("entrypoints.cu")).compile("ferrule_cuda_core");
    configure_implementation(cutlass_root.join("entrypoints.cu")).compile("ferrule_cuda_cutlass");
}

fn publish_cuda_driver_search_path() {
    let wsl_driver_dir = Path::new("/usr/lib/wsl/lib");
    if wsl_driver_dir.join("libcuda.so").is_file() {
        println!(
            "cargo:rustc-link-search=native={}",
            wsl_driver_dir.display()
        );
    }
}

fn native_codegen_target(target: CudaTarget) -> String {
    match target.compute_capability() {
        // These CUTLASS collectives require the complete architecture-specific
        // instruction feature set. This is an NVCC code-generation scope, not a
        // different device architecture; the detected and published target keeps
        // the exact compute capability reported by the device.
        compute_capability @ (90 | 100 | 101 | 103 | 110) => {
            format!("sm_{compute_capability}a")
        }
        compute_capability => format!("sm_{compute_capability}"),
    }
}

fn resolve_cuda_arch() -> String {
    if let Ok(target) = env::var(CUDA_ARCH_ENV) {
        let target = target.trim();
        if !target.is_empty() {
            return target.to_owned();
        }
    }

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"])
        .output()
        .unwrap_or_else(|error| {
            panic!("{CUDA_ARCH_ENV} is unset and nvidia-smi could not be executed: {error}")
        });
    if !output.status.success() {
        panic!(
            "{CUDA_ARCH_ENV} is unset and nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let capability =
        String::from_utf8(output.stdout).expect("nvidia-smi compute capability must be UTF-8");
    let capability = capability.lines().next().unwrap_or_default().trim();
    let digits = capability.replace('.', "");
    if digits.len() < 2 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        panic!("nvidia-smi returned invalid compute capability `{capability}`");
    }
    format!("sm_{digits}")
}

fn verify_nvcc_support(cuda_arch: &str, target: CudaTarget) {
    let output = Command::new("nvcc")
        .arg("--list-gpu-code")
        .output()
        .unwrap_or_else(|error| panic!("failed to execute nvcc: {error}"));
    if !output.status.success() {
        panic!(
            "nvcc --list-gpu-code failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let base = format!("sm_{}", target.compute_capability());
    let codes = String::from_utf8_lossy(&output.stdout);
    if !codes.lines().any(|line| line.trim() == base) {
        panic!("nvcc does not publish base code `{base}` required by `{cuda_arch}`");
    }
}

fn require_cutlass_headers(cutlass_dir: &Path) {
    let version_header = cutlass_dir.join("include/cutlass/version.h");
    if !version_header.is_file() {
        panic!(
            "CUTLASS headers were not found at {}. Run `just cutlass-setup` or set {CUTLASS_DIR_ENV} to the pinned checkout",
            cutlass_dir.display()
        );
    }
    println!("cargo:rerun-if-changed={}", version_header.display());
}
