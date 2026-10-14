//! CPU-only admission and native overflow policy; never initializes CUDA.
use ferrule_backend::cuda_plan::bf16_gemmex::*;
use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};
use std::{
    io::Write,
    process::{Command, Stdio},
};
fn layout(n: usize, k: usize) -> NumericFp8Layout {
    NumericFp8Layout {
        n,
        k,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    }
}
#[test]
fn budgets_are_explicit_disjoint_and_shape_general() {
    for m in [1, 2, 5, 16, 32] {
        for (n, k) in [(128, 128), (1024, 2048), (2048, 512), (8192, 2048)] {
            let p = Bf16GemmExPlan::new(layout(n, k), m, usize::MAX, BF16_GEMMEX_PROVIDER_BYTES)
                .unwrap();
            assert_eq!(p.staging_bytes(), ((m + n) * k + m * n) * 2);
            assert_eq!(p.storage_bytes(), p.staging_bytes() + 32 * 1024 * 1024);
            assert_eq!(p.provider_bytes(), 16 * 1024 * 1024);
            assert!(
                Bf16GemmExPlan::new(
                    layout(n, k),
                    m,
                    p.storage_grant_bytes() - 1,
                    p.provider_bytes()
                )
                .is_err()
            );
            assert!(
                Bf16GemmExPlan::new(
                    layout(n, k),
                    m,
                    p.storage_grant_bytes(),
                    p.provider_bytes() - 1
                )
                .is_err()
            );
            assert!(
                Bf16GemmExPlan::new(layout(n, k), m, p.storage_grant_bytes(), p.provider_bytes())
                    .is_ok()
            );
        }
    }
}
#[test]
fn rejects_unsupported_layout_and_overflow() {
    for (m, n, k) in [
        (0, 128, 128),
        (33, 128, 128),
        (1, 0, 128),
        (1, 128, 0),
        (1, 129, 128),
        (1, 128, 129),
        (1, usize::MAX, 128),
        (1, 128, usize::MAX),
    ] {
        assert!(Bf16GemmExPlan::new(layout(n, k), m, usize::MAX, usize::MAX).is_err());
    }
    for l in [
        NumericFp8Layout {
            row_origin: 128,
            ..layout(128, 128)
        },
        NumericFp8Layout {
            column_origin: 1,
            ..layout(128, 128)
        },
        NumericFp8Layout {
            scale_type: NumericFp8ScaleType::F32,
            ..layout(128, 128)
        },
    ] {
        assert!(Bf16GemmExPlan::new(l, 1, usize::MAX, usize::MAX).is_err());
    }
}
#[test]
fn native_policy_checks_alignment_overflow_and_alias() {
    let source = r#"
#include "cutlass/bf16_gemmex_policy.h"
using namespace ferrule::cuda::cutlass::bf16_gemmex;
static_assert(geometry(1,2048,512));
static_assert(geometry(5,8192,2048));
static_assert(!geometry(33,128,128));
static_assert(!geometry(1,129,128));
static_assert(required(1,2048,512)==((1ull+2048)*512+2048)*2+(32ull<<20));
static_assert(storage(1,2048,512,0x1000,required(1,2048,512)));
static_assert(!storage(1,2048,512,0x1001,required(1,2048,512)));
static_assert(!storage(1,2048,512,0x1000,required(1,2048,512)-1));
static_assert(!storage(1,2048,512,UINT64_MAX & ~255ull,UINT64_MAX));
static_assert(!span(0,256,256));
static_assert(!span(0x1000,UINT64_MAX,256));
static_assert(overlap(256,256,256,128));
static_assert(overlap(256,256,384,256));
static_assert(!overlap(256,256,512,128));
"#;
    let mut p = Command::new("c++")
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
    p.stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
