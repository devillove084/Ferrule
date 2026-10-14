fn main() {
    println!("cargo:rerun-if-env-changed=FERRULE_NVTX_INCLUDE_DIR");
    println!("cargo:rerun-if-changed=native/nvtx.c");
    #[cfg(feature = "nvtx-profile")]
    build_nvtx();
}

#[cfg(feature = "nvtx-profile")]
fn build_nvtx() {
    let include = std::env::var_os("FERRULE_NVTX_INCLUDE_DIR")
        .map(std::path::PathBuf::from)
        .expect("nvtx-profile requires FERRULE_NVTX_INCLUDE_DIR containing nvtx3/nvToolsExt.h");
    assert!(
        include.join("nvtx3/nvToolsExt.h").is_file(),
        "FERRULE_NVTX_INCLUDE_DIR must contain the official NVTX3 nvtx3/nvToolsExt.h"
    );
    println!("cargo:rerun-if-changed={}", include.join("nvtx3").display());
    cc::Build::new()
        .file("native/nvtx.c")
        .include(include)
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("ferrule_nvtx_host");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=dl");
    }
}
