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
static int bf16_available = 1;
static int m1_available = 1;
static int w8a16_available = 0;
static int gemv_available = 0;
extern "C" int32_t ferrule_numeric_bf16_gemv_available(void) { return gemv_available; }
extern "C" int32_t ferrule_cutlass_numeric_fp8_w8a16_available(void) { return w8a16_available; }
extern "C" int32_t ferrule_cutlass_bf16_available(void) { return bf16_available; }
extern "C" int32_t ferrule_cutlass_f32_available(void) { return f32_available; }
extern "C" int32_t ferrule_cutlass_numeric_fp8_m1_available(void) { return m1_available; }
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
        ("BF16_GEMM", CutlassKernelId::Bf16Gemm),
        ("NUMERIC_BF16_M1_GEMV", CutlassKernelId::NumericBf16M1Gemv),
        ("NUMERIC_FP8_FUSED_M1", CutlassKernelId::NumericFp8FusedM1),
        (
            "NUMERIC_FP8_W8A16_FUSED",
            CutlassKernelId::NumericFp8W8A16Fused,
        ),
    ] {
        code.push_str(&format!(
            "static_assert(FERRULE_CUTLASS_KERNEL_{name} == {}u);\nstatic_assert(FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_{name}) == {}ull);\n",
            kernel as u32, kernel.mask(),
        ));
    }
    code.push_str(
        r#"
int main() {
  for (int w8 = 0; w8 <= 2; ++w8) {
    w8a16_available = w8;
    const uint64_t w8_bit = w8 == 1
        ? FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_NUMERIC_FP8_W8A16_FUSED)
        : 0ull;
    f32_available = bf16_available = m1_available = 1;
    assert(ferrule_cutlass_provider_manifest().kernel_mask == (0x1d86ull | w8_bit));
    // F32, BF16, M1 and W8A16 availability are independently reported. A
    // non-success W8 return cannot accidentally advertise its new native bit.
    f32_available = 0;
    assert(ferrule_cutlass_provider_manifest().kernel_mask == (0x1986ull | w8_bit));
    bf16_available = 0;
    assert(ferrule_cutlass_provider_manifest().kernel_mask == (0x1186ull | w8_bit));
    f32_available = 1;
    assert(ferrule_cutlass_provider_manifest().kernel_mask == (0x1586ull | w8_bit));
    m1_available = 0;
    assert(ferrule_cutlass_provider_manifest().kernel_mask == (0x586ull | w8_bit));
    for (int gemv = 0; gemv <= 2; ++gemv) {
      gemv_available = gemv;
      const uint64_t bit = gemv == 1
          ? FERRULE_CUTLASS_KERNEL_BIT(FERRULE_CUTLASS_KERNEL_NUMERIC_BF16_M1_GEMV)
          : 0ull;
      assert(ferrule_cutlass_provider_manifest().kernel_mask == (0x586ull | w8_bit | bit));
    }
    gemv_available = 0;
  }
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
    let (production, oracle) = split_oracle_builder();
    // Two *production* CUDA TUs share one single-source builder closure; the
    // other production builder is CPU-only. Do not just relax the global count.
    assert_eq!(production.matches("cc::Build::new()").count(), 2);
    assert_eq!(production.matches(".cuda(true)").count(), 1);
    assert_eq!(production.matches(".file(source)").count(), 1);
    assert_eq!(production.matches(".file(").count(), 2);
    assert_eq!(production.matches("configure_provider(").count(), 2);
    assert_eq!(production.matches(".compile(\"ferrule_cuda_").count(), 2);
    assert_eq!(production.matches(".compile(").count(), 3); // two CUDA + CPU
    assert_eq!(production.matches(".include(&native_root)").count(), 2);
    assert_eq!(production.matches(".include(").count(), 2);
    assert!(!production.contains(".join(\"tests/numeric_w8a16/reference.cu\")"));
    assert!(!production.contains("ferrule_test_w8a16_scalar"));
    assert_eq!(oracle.matches("cc::Build::new()").count(), 1);
    assert_eq!(oracle.matches(".cuda(true)").count(), 1);
    assert_eq!(oracle.matches(".file(").count(), 1);
    assert_eq!(oracle.matches(".compile(").count(), 1);
    assert!(oracle.contains(".file(manifest_dir.join(\"tests/numeric_w8a16/reference.cu\"))"));
    assert!(oracle.contains(".compile(\"ferrule_test_w8a16_scalar\")"));
    assert!(!oracle.contains(".include("));
    assert!(!oracle.contains("configure_provider("));

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

/// Isolate precisely the oracle build branch after the two production compiles,
/// not the earlier cfg/define branches. Keep all trailing code in production so
/// an accidental extra unconditional TU/link cannot disappear from the checks.
fn split_oracle_builder() -> (String, &'static str) {
    let anchor = "configure_provider(native_root.join(\"cutlass/provider.cu\")).compile(\"ferrule_cuda_cutlass\");";
    let (prefix, suffix) = BUILD_RS.split_once(anchor).unwrap();
    let suffix = suffix
        .trim_start()
        .strip_prefix("if cuda_test_oracle {")
        .expect("third CUDA TU must be inside the explicit oracle gate");
    let (oracle, suffix) = suffix
        .split_once("\n    }\n")
        .expect("oracle gate must close before subsequent build logic");
    (format!("{prefix}{anchor}{suffix}"), oracle)
}

#[test]
fn scalar_oracle_is_explicit_test_only_and_not_a_production_dependency() {
    assert!(BUILD_RS.contains("const CUDA_TEST_ORACLE_ENV: &str = \"FERRULE_CUDA_TEST_ORACLE\";"));
    assert!(BUILD_RS.contains(
        "let cuda_test_oracle = env::var(CUDA_TEST_ORACLE_ENV).is_ok_and(|value| value == \"1\");"
    ));
    assert!(BUILD_RS.contains("if cuda_test_oracle {\n        println!(\"cargo:rustc-cfg=ferrule_cuda_test_oracle\");\n    }"));
    assert!(BUILD_RS.contains("if cuda_test_oracle {\n            build.define(\"FERRULE_CUDA_TEST_ORACLE\", Some(\"1\"));\n        }"));
    assert_eq!(
        BUILD_RS
            .matches("rustc-cfg=ferrule_cuda_test_oracle")
            .count(),
        1
    );
    assert_eq!(
        BUILD_RS
            .matches("build.define(\"FERRULE_CUDA_TEST_ORACLE\"")
            .count(),
        1
    );
    let (_, oracle) = split_oracle_builder();
    for strict in [".flag(\"--ftz=false\")", ".flag(\"--fmad=false\")"] {
        assert!(oracle.contains(strict));
    }
    assert!(!oracle.contains("fast_math"));
    assert!(!oracle.contains(".define("));
    let reference = include_str!("numeric_w8a16/reference.cu");
    let includes: Vec<_> = reference
        .lines()
        .filter(|line| line.starts_with("#include"))
        .collect();
    assert_eq!(
        includes,
        [
            "#include <cuda_runtime.h>",
            "#include <cmath>",
            "#include <cstdint>"
        ]
    );
    let fixture = include_str!("numeric_w8a16.rs");
    assert!(fixture.contains("#[cfg(all(feature = \"cuda\", ferrule_cuda_test_oracle))]"));
    // No production Rust FFI, native header, or provider may reference the
    // standalone scalar oracle. Existing gated production diagnostics are not
    // scalar-reference implementations and keep their original ownership.
    fn check_tree(path: &std::path::Path) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                check_tree(&path);
            } else if path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| matches!(s, "rs" | "cu" | "cuh" | "h"))
            {
                let text = std::fs::read_to_string(&path).unwrap();
                for forbidden in ["ferrule_test_w8a16_scalar", "numeric_w8a16/reference.cu"] {
                    assert!(
                        !text.contains(forbidden),
                        "production dependency on test oracle: {}",
                        path.display()
                    );
                }
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    check_tree(&root.join("src"));
    check_tree(&root.join("native"));
}

#[test]
fn core_provider_is_an_include_only_translation_unit() {
    assert!(CORE_PROVIDER.lines().all(|line| {
        let line = line.trim();
        line.is_empty() || line.starts_with("#include \"core/")
    }));
    for header in ["bindings.cuh", "recurrent_bf16.cuh", "bf16_boundary.cuh"] {
        assert!(
            CORE_PROVIDER
                .lines()
                .any(|line| line.trim() == format!("#include \"core/{header}\""))
        );
    }
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
