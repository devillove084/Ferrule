//! Host-only policy checks: no CUDA headers, driver initialization or device calls.
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn gemv_native_policy_pins_runtime_and_bounds_full_warmup_scratch() {
    let source = format!(
        r#"
#include "cutlass/numeric_bf16_gemv_policy.h"
using namespace ferrule::cuda::cutlass::numeric_bf16_gemv;
static_assert(supported_build(86, 12, 4));
static_assert(!supported_build(75, 12, 4));
static_assert(!supported_build(86, 11, 8));
static_assert(!supported_build(86, 12, 3));
static_assert(!supported_build(86, 12, 9));
static_assert(!supported_build(86, 13, 4));
static_assert(supported_runtime(120405));
static_assert(!supported_runtime(120402));
static_assert(!supported_runtime(120404));
static_assert(!supported_runtime(120406));
static_assert(!supported_runtime(120901));
static_assert(!supported_runtime(130401));
static_assert(kScratch == {scratch});
static_assert(kTile == {tile});
constexpr uint64_t last = (UINT64_MAX - kScratch) & ~uint64_t(15);
static_assert(valid_scratch(last, kScratch));
static_assert(valid_scratch(last, UINT64_MAX));
static_assert(!valid_scratch(last + 16, kScratch));
static_assert(!valid_scratch((UINT64_MAX - kStaging) & ~uint64_t(15), kScratch));
static_assert(!valid_scratch(0, kScratch));
static_assert(!valid_scratch(0x1001, kScratch));
static_assert(!valid_scratch(0x1000, kScratch - 1));
"#,
        scratch = ferrule_backend::cuda_plan::NUMERIC_BF16_GEMV_SCRATCH_BYTES,
        tile = ferrule_backend::cuda_plan::NUMERIC_BF16_GEMV_TILE_ROWS,
    );
    let mut child = Command::new("c++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I",
        ])
        .arg(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("native/cuda"))
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[path = "numeric_bf16_gemv/pair_swiglu_direct_tail.rs"]
mod pair_swiglu_direct_tail;
