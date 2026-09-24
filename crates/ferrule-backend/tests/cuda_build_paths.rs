const BUILD_RS: &str = include_str!("../build.rs");
const CORE_PROVIDER: &str = include_str!("../native/cuda/core/provider.cu");
const CUTLASS_PROVIDER: &str = include_str!("../native/cuda/cutlass/provider.cu");

#[cfg(feature = "cuda")]
#[test]
fn cutlass_native_abi_ids_and_unavailable_f32_manifest() {
    use ferrule_backend::cuda::providers::cutlass::CutlassKernelId;
    use std::{fs, process::Command};
    let directory =
        std::env::temp_dir().join(format!("ferrule-cutlass-manifest-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let source = directory.join("manifest.cc");
    let binary = directory.join("manifest-test");
    let mut code = String::from(
        r#"
#include "cutlass/manifest.cuh"
#include <cassert>
static int f32_available = 1;
extern "C" int32_t ferrule_cutlass_f32_available(void) { return f32_available; }
static_assert(sizeof(FerruleCutlassProviderManifest) == 8);
static_assert(alignof(FerruleCutlassProviderManifest) == 8);
"#,
    );
    for (name, kernel) in [
        ("FP8_QUERY_A_KV", CutlassKernelId::Fp8QueryAKv),
        ("BF16_COMPRESSOR", CutlassKernelId::Bf16Compressor),
        (
            "HYPER_CONNECTION_PRODUCER",
            CutlassKernelId::HyperConnectionProducer,
        ),
        ("SHARED_FFN", CutlassKernelId::SharedFfn),
        ("GROUPED_FP4_MOE", CutlassKernelId::GroupedFp4Moe),
        ("MLA_OUTPUT", CutlassKernelId::MlaOutput),
        ("MAIN_PROJECT_NORM", CutlassKernelId::MainProjectNorm),
        ("HYBRID_MLA_ATTENTION", CutlassKernelId::HybridMlaAttention),
        ("PROPOSAL_HEAD", CutlassKernelId::ProposalHead),
        ("FP8_PROJECTION", CutlassKernelId::Fp8Projection),
        ("F32_GEMM", CutlassKernelId::F32Gemm),
    ] {
        code.push_str(&format!(
            "static_assert(FERRULE_CUTLASS_KERNEL_{name} == {}u);\nstatic_assert(FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_{name}) == {}ull);\n",
            kernel as u32, kernel.mask(),
        ));
    }
    code.push_str(
        r#"
int main() {
  assert(ferrule_cutlass_provider_manifest().kernel_mask == 0x586ull);
  // Model a build with no compiled F32 TensorOp. Other native bits must stay
  // identical; neither Rust nor the native manifest may infer F32 from BF16.
  f32_available = 0;
  assert(ferrule_cutlass_provider_manifest().kernel_mask == 0x186ull);
}
"#,
    );
    fs::write(&source, code).unwrap();
    let output = Command::new("c++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DFERRULE_CUDA_TARGET_SM=86",
            "-DFERRULE_CUDA_HAS_BASELINE_SIMT=1",
            "-DFERRULE_CUDA_HAS_BF16_MMA_SYNC=1",
            "-DFERRULE_CUDA_HAS_FP8_MMA_SYNC=0",
            "-DFERRULE_CUDA_HAS_SM90_WGMMA=0",
            "-DFERRULE_CUDA_HAS_SM1XX_UMMA=0",
            "-DFERRULE_CUDA_HAS_SM103_BLOCK_SCALED_FP4=0",
            "-DFERRULE_CUDA_HAS_SM12X_MXFP4_MMA_SYNC=0",
        ])
        .arg("-I")
        .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("native/cuda"))
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(Command::new(&binary).status().unwrap().success());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn cuda_build_uses_canonical_provider_sources() {
    assert_eq!(
        BUILD_RS
            .matches("configure_provider(native_root.join(")
            .count(),
        2
    );
    assert!(BUILD_RS.contains("native_root.join(\"core/provider.cu\")"));
    assert!(BUILD_RS.contains("native_root.join(\"cutlass/provider.cu\")"));
    for forbidden in [
        "entrypoints.cu",
        "implementations/",
        "native/cuda/implementations",
        "operators_root",
        concat!("port", "able_root"),
        "cutlass_root",
        "providers_root",
    ] {
        assert!(
            !BUILD_RS.contains(forbidden),
            "build.rs must not reference a legacy CUDA build path: {forbidden}"
        );
    }
}

#[test]
fn each_cuda_provider_is_one_non_rdc_compilation_unit() {
    assert_eq!(BUILD_RS.matches("cc::Build::new()").count(), 2);
    assert_eq!(BUILD_RS.matches(".file(source)").count(), 1);
    assert_eq!(BUILD_RS.matches("configure_provider(").count(), 2);
    assert_eq!(BUILD_RS.matches(".include(&native_root)").count(), 2);
    assert_eq!(BUILD_RS.matches(".include(").count(), 2);

    for forbidden in [
        ".files(",
        ".rdc(",
        "-rdc=",
        "--relocatable-device-code",
        "--device-link",
    ] {
        assert!(
            !BUILD_RS.contains(forbidden),
            "build.rs must not enable multi-source or RDC mode: {forbidden}"
        );
    }
}

#[test]
fn core_provider_is_an_include_only_translation_unit() {
    assert_eq!(CORE_PROVIDER.trim(), "#include \"core/bindings.cuh\"");
}

#[test]
fn cutlass_provider_is_an_include_only_translation_unit() {
    assert!(CUTLASS_PROVIDER.lines().all(|line| {
        let line = line.trim();
        line.is_empty() || line.starts_with("#include \"cutlass/")
    }));
    assert!(CUTLASS_PROVIDER.contains("cutlass/abi_checks.cuh"));
    assert!(CUTLASS_PROVIDER.contains("cutlass/manifest.cuh"));
    assert!(
        CUTLASS_PROVIDER
            .lines()
            .any(|line| line.trim() == "#include \"cutlass/bindings.cuh\"")
    );
}
