const BUILD_RS: &str = include_str!("../build.rs");
const CORE_PROVIDER: &str = include_str!("../native/cuda/core/provider.cu");
const CUTLASS_PROVIDER: &str = include_str!("../native/cuda/cutlass/provider.cu");

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
