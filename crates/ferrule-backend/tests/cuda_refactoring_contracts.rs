//! CPU-required source contracts supplement (not replace) runtime fault tests.
const CONTEXT: &str = include_str!("../src/cuda/context.rs");
const GRAPH: &str = include_str!("../src/cuda/graph.rs");

fn function<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}")).expect(name);
    let rest = &source[start..];
    let end = ["\n    fn ", "\n    pub fn "]
        .iter()
        .filter_map(|s| rest.find(s))
        .min()
        .unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn pr01_wrappers_defer_native_operations() {
    assert!(
        function(CONTEXT, "record_device_allocation").contains("FnOnce"),
        "allocation wrapper evaluates Result before precheck"
    );
    assert!(
        function(CONTEXT, "record_stream_wide_sync").contains("FnOnce"),
        "sync wrapper evaluates Result before precheck"
    );
    assert!(!CONTEXT.contains("record_stream_wide_sync(self."));
}

#[test]
fn pr01_omitted_entries_and_submitted_cleanup_are_guarded() {
    assert!(function(CONTEXT, "download_i32_buffer").contains("check_capture_safe"));
    assert!(function(CONTEXT, "alloc_managed_u8_len").contains("record_device_allocation"));
    for name in ["i32_host_mirror", "dsv4_router_token_ids"] {
        let body = function(CONTEXT, name);
        assert!(
            body.find("check_capture_safe").unwrap()
                < body.find("PinnedHostBuffer::from_slice").unwrap()
        );
    }
    assert!(CONTEXT.contains("submitted_copy_cleanup"));
}

#[test]
fn pr24_capture_is_scoped_and_key_keeps_boundaries() {
    assert!(
        GRAPH.contains("CaptureGuard"),
        "capture must own exactly one end on Err/unwind"
    );
    assert!(
        !GRAPH.contains(".chain(shapes.iter().copied())"),
        "pointer/shape boundary collision"
    );
    assert!(
        GRAPH.contains("Unknown"),
        "unknown completion must not remain replayable"
    );
}

// Compile the real CPU-only counter implementation even without the CUDA feature.
#[allow(dead_code)]
#[path = "../src/cuda/counters.rs"]
mod counters;

// Exercise the actual guard/key/cache protocol without linking CUDA. Native
// lifecycle tests remain in the CUDA-enabled library and required GPU profile.
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
#[allow(unsafe_code)]
#[path = "../src/cuda/graph.rs"]
mod graph;

const LINEAR: &str = include_str!("../src/cuda/operators/linear.rs");
const NUMERIC_BRIDGE: &str = include_str!("../src/cuda/operators/numeric_fp8.rs");
const NUMERIC_CONTRACT: &str = include_str!("../src/cuda/operators/numeric.rs");
const NUMERIC_PROVIDER: &str = include_str!("../src/cuda/providers/cutlass_bf16.rs");
const CUTLASS: &str = include_str!("../src/cuda/providers/cutlass.rs");

fn body(source: &str, name: &str) -> String {
    let start = source
        .find(&format!("fn {name}("))
        .or_else(|| source.find(&format!("fn {name}<")))
        .expect(name);
    let rest = &source[start..];
    let open = rest.find('{').unwrap();
    let mut depth = 0;
    for (index, ch) in rest[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return compact(&rest[open + 1..open + index]);
                }
            }
            _ => {}
        }
    }
    panic!("unclosed function {name}");
}

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn pr21_linear_facade_delegates_without_changing_execution() {
    // Exact bodies rule out extra preflight, allocation, sync, fallback, error
    // conversion, or poison mutation in the new dispatch layer.
    for (name, arguments) in [
        ("bf16_gemm_workspace_requirements", "layout"),
        (
            "bf16_gemm_can_implement",
            "stream,activation,weight,output,layout",
        ),
        ("bf16_gemm", "stream,activation,weight,output,layout"),
        (
            "numeric_fp8_linear",
            "stream,artifact,activation,output,workspace,plan,activation_stride,output_stride",
        ),
    ] {
        let actual = body(LINEAR, name).replace(",)", ")");
        assert_eq!(actual, format!("cutlass::{name}({arguments})"), "{name}");
    }
    assert!(LINEAR.contains("pub struct Bf16GemmLayout"));
    assert!(!NUMERIC_PROVIDER.contains("pub struct Bf16GemmLayout"));
    assert_eq!(body(LINEAR, "validate"), "validate_bf16_gemm_layout(self)");
    assert!(NUMERIC_BRIDGE.contains("use crate::cuda::operators::linear::{"));
    assert!(!NUMERIC_BRIDGE.contains("providers::cutlass"));
}

#[test]
fn pr21_numeric_preflight_and_native_failure_keep_poison_boundary() {
    let launch = body(NUMERIC_PROVIDER, "numeric_fp8_linear");
    let checkpoints = [
        "ifworkspace.is_poisoned()",
        "ifartifact.layout()!=plan.layout()",
        "ifworkspace.precision()!=plan.precision()",
        "available(plan.precision())?;",
        "no_overlap(&[a,w,s],&[d,scratch])?;",
        "status(unsafe{preflight(&args)})?;",
        "cu(stream.context().bind_to_thread())?;",
        "letresult=status(unsafe{launch(&args)});",
        "ifresult.is_err(){workspace.poison();}",
    ];
    let mut remaining = launch.as_str();
    for checkpoint in checkpoints {
        let index = remaining.find(checkpoint).expect(checkpoint);
        remaining = &remaining[index + checkpoint.len()..];
    }
    assert_eq!(remaining, "result");
    assert_eq!(launch.matches("workspace.poison();").count(), 1);
    let status = body(NUMERIC_PROVIDER, "status");
    assert!(status.contains("0=>Ok(())"));
    assert!(status.contains("nativesubmissionfailed({code});completionunknown"));
    assert!(body(NUMERIC_PROVIDER, "buffer").contains("b.check_context(stream.context()"));
    assert!(launch.contains("workspace.allocated_bytes()>plan.scratch_budget_bytes()"));
}

#[test]
fn pr21_compatibility_exports_have_explicit_migration_boundaries() {
    assert!(
        CUTLASS
            .contains("pub type Bf16GemmLayout = crate::cuda::operators::linear::Bf16GemmLayout;")
    );
    assert!(CUTLASS.contains("#[deprecated("));
    assert!(CUTLASS.contains("provider paths are compatibility-only"));
    assert!(CUTLASS.contains("Compatibility paths for operator-owned numeric FP8 contracts"));
    assert!(
        include_str!("../src/cuda/providers/cutlass_legacy.rs")
            .contains("pub trait NumericFp8PrecisionExt")
    );
    assert!(NUMERIC_CONTRACT.contains("Compatibility raw-storage constructor"));
}

#[cfg(feature = "cuda")]
#[test]
#[allow(deprecated)]
fn pr21_semantic_layout_preserves_legacy_type_and_validation() {
    use ferrule_backend::cuda::operators::linear::Bf16GemmLayout;
    use ferrule_backend::cuda::providers::cutlass;

    // No GPU needed: both paths retain struct literals, methods and type identity.
    let semantic = Bf16GemmLayout::contiguous(3, 5, 8);
    let legacy: cutlass::Bf16GemmLayout = semantic;
    let roundtrip: Bf16GemmLayout = legacy;
    assert_eq!(roundtrip, semantic);
    semantic.validate().unwrap();
    for invalid in [
        Bf16GemmLayout {
            rows: 0,
            ..semantic
        },
        Bf16GemmLayout { k: 7, ..semantic },
        Bf16GemmLayout {
            activation_stride: 7,
            ..semantic
        },
        Bf16GemmLayout {
            output_stride: 4,
            ..semantic
        },
    ] {
        assert!(invalid.validate().is_err());
    }
    let _: fn(
        Bf16GemmLayout,
    ) -> ferrule_common::Result<
        ferrule_backend::cuda::operators::OperatorWorkspaceRequirements,
    > = ferrule_backend::cuda::operators::linear::bf16_gemm_workspace_requirements;
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA GPU with BF16 TensorOp support"]
fn pr21_bf16_facade_workspace_preflight_and_launch() {
    use ferrule_backend::cuda::operators::linear::{
        Bf16GemmLayout, bf16_gemm, bf16_gemm_can_implement, bf16_gemm_workspace_requirements,
    };
    use ferrule_backend::cuda::providers::{CudaContext, DeviceBuffer};

    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let other_context = CudaContext::new(0).unwrap();
    let other_stream = other_context.new_stream().unwrap();
    let activation = DeviceBuffer::from_host(&stream, &[0x3f80u16; 8]).unwrap();
    let weight = DeviceBuffer::from_host(&stream, &[0x3f80u16; 8]).unwrap();
    let foreign = DeviceBuffer::from_host(&other_stream, &[0x3f80u16; 8]).unwrap();
    let mut output = DeviceBuffer::from_host(&stream, &[73.0f32]).unwrap();
    let layout = Bf16GemmLayout::contiguous(1, 1, 8);
    let requirements = bf16_gemm_workspace_requirements(layout).unwrap();
    assert_eq!(requirements.bytes, 0);
    assert_eq!(requirements.alignment, 1);

    bf16_gemm_can_implement(&stream, &activation, &weight, &output, layout).unwrap();
    assert_eq!(output.to_host_vec(&stream).unwrap(), [73.0]);
    assert!(bf16_gemm_can_implement(&stream, &foreign, &weight, &output, layout).is_err());
    assert!(bf16_gemm(&stream, &foreign, &weight, &mut output, layout).is_err());
    let invalid = Bf16GemmLayout { k: 7, ..layout };
    assert!(bf16_gemm_workspace_requirements(invalid).is_err());
    assert!(bf16_gemm(&stream, &activation, &weight, &mut output, invalid).is_err());
    assert_eq!(output.to_host_vec(&stream).unwrap(), [73.0]);

    bf16_gemm(&stream, &activation, &weight, &mut output, layout).unwrap();
    assert_eq!(output.to_host_vec(&stream).unwrap(), [8.0]);
}

#[test]
fn pr21_numeric_contracts_are_operator_owned_not_provider_forwarders() {
    for declaration in [
        "pub enum NumericFp8Precision",
        "pub struct NumericFp8LinearPlan",
        "pub struct CudaNumericFp8Artifact",
        "pub struct CudaNumericFp8Workspace",
    ] {
        assert!(NUMERIC_CONTRACT.contains(declaration), "{declaration}");
        assert!(!NUMERIC_PROVIDER.contains(declaration), "{declaration}");
    }
    // Layout is the shared, provider-neutral storage contract used by validated
    // CPU payloads as well. Do not fork it or wrap it just for CUDA.
    assert!(NUMERIC_CONTRACT.contains("pub use ferrule_common::numeric_fp8::{"));
    let implementation = NUMERIC_CONTRACT
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!implementation.contains("providers::"));
    assert!(!NUMERIC_CONTRACT.contains("CutlassKernelId"));
    assert!(!NUMERIC_CONTRACT.contains("pub const fn kernel("));
    assert!(!LINEAR.contains("pub use crate::cuda::providers::cutlass::bf16_linear"));
    assert!(!LINEAR.contains("cutlass::bf16_linear::validate"));
    assert!(NUMERIC_PROVIDER.contains("fn kernel_for_precision("));
    assert!(CUTLASS.contains("pub use crate::cuda::operators::linear::{"));
    assert!(!CUTLASS.contains("pub(crate) mod bf16_linear"));
    for private_field in ["storage: DeviceBuffer<u8>", "poisoned: bool"] {
        assert!(NUMERIC_CONTRACT.contains(private_field));
        assert!(!NUMERIC_CONTRACT.contains(&format!("pub(crate) {private_field}")));
        assert!(!NUMERIC_CONTRACT.contains(&format!("pub {private_field}")));
    }
    assert!(NUMERIC_CONTRACT.contains("pub(crate) fn storage("));
    assert!(NUMERIC_CONTRACT.contains("pub(crate) fn poison("));
    assert_eq!(body(NUMERIC_CONTRACT, "poison"), "self.poisoned=true;");
}

#[cfg(feature = "cuda")]
#[test]
#[allow(deprecated)]
fn pr21_numeric_facade_and_legacy_exports_share_semantic_identity() {
    use ferrule_backend::cuda::operators::linear as semantic;
    use ferrule_backend::cuda::providers::cutlass as legacy;
    use std::any::{TypeId, type_name};

    macro_rules! same_type {
        ($ty:ident) => {
            assert_eq!(TypeId::of::<semantic::$ty>(), TypeId::of::<legacy::$ty>());
            assert!(type_name::<semantic::$ty>().contains("::operators::linear::numeric::"));
        };
    }
    same_type!(NumericFp8Precision);
    same_type!(NumericFp8LinearPlan);
    same_type!(CudaNumericFp8Artifact);
    same_type!(CudaNumericFp8Workspace);
    assert_eq!(
        TypeId::of::<semantic::NumericFp8Layout>(),
        TypeId::of::<legacy::NumericFp8Layout>()
    );
    assert_eq!(
        TypeId::of::<semantic::NumericFp8Layout>(),
        TypeId::of::<ferrule_backend::numeric_fp8::NumericFp8Layout>()
    );

    let layout = semantic::NumericFp8Layout {
        n: 259,
        k: 259,
        row_origin: 127,
        column_origin: 125,
        scale_type: semantic::NumericFp8ScaleType::F32,
    };
    for (precision, budget, activation, weights, launches) in [
        (
            semantic::NumericFp8Precision::Bf16RneF32Accumulate,
            10567,
            1584,
            8976,
            33,
        ),
        (
            semantic::NumericFp8Precision::F32Tf32x3,
            17619,
            0,
            17612,
            32,
        ),
    ] {
        let plan = semantic::NumericFp8LinearPlan::new(layout, 3, budget, precision).unwrap();
        let legacy: legacy::NumericFp8LinearPlan = plan;
        assert_eq!(plan, legacy);
        assert_eq!(plan.layout(), layout);
        assert_eq!(plan.precision(), precision);
        assert_eq!(plan.scratch_budget_bytes(), budget);
        assert_eq!(plan.activation_bytes(), activation);
        assert_eq!(plan.weight_tile_bytes(), weights);
        assert_eq!(
            plan.workspace_requirements().bytes as usize,
            activation + weights
        );
        assert_eq!(plan.workspace_requirements().alignment, 16);
        assert_eq!(plan.tile_rows(), 17);
        assert_eq!(plan.tile_count(), 16);
        assert_eq!(plan.kernel_launches(), launches);
        assert!(semantic::NumericFp8LinearPlan::new(layout, 0, budget, precision).is_err());
        assert!(semantic::NumericFp8LinearPlan::new(layout, 3, 0, precision).is_err());
    }

    // Both the original inherent const API and the intermediate trait survive.
    const F32_KERNEL: legacy::CutlassKernelId = semantic::NumericFp8Precision::F32Tf32x3.kernel();
    const BF16_KERNEL: legacy::CutlassKernelId =
        legacy::NumericFp8Precision::Bf16RneF32Accumulate.kernel();
    assert_eq!(F32_KERNEL, legacy::CutlassKernelId::F32Gemm);
    assert_eq!(BF16_KERNEL, legacy::CutlassKernelId::Bf16Gemm);
    assert_eq!(
        legacy::NumericFp8PrecisionExt::kernel(semantic::NumericFp8Precision::F32Tf32x3),
        F32_KERNEL
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA sm80+; semantic owner API for both numeric profiles"]
fn pr21_numeric_owner_facade_upload_budget_and_submission() {
    use ferrule_backend::cuda::operators::linear::{
        CudaOperators, NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision,
        NumericFp8ScaleType,
    };
    let op = CudaOperators::new_on_device(0).unwrap();
    let layout = NumericFp8Layout {
        n: 3,
        k: 3,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::F32,
    };
    let artifact = op
        .upload_numeric_fp8_linear(layout, &[0x38; 9], &1.0f32.to_le_bytes())
        .unwrap();
    assert_eq!(artifact.layout(), layout);
    assert_eq!(artifact.storage_bytes(), 13);
    let input = op
        .upload_f32_buffer(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    for (precision, budget) in [
        (NumericFp8Precision::Bf16RneF32Accumulate, 64),
        (NumericFp8Precision::F32Tf32x3, 24),
    ] {
        let plan = NumericFp8LinearPlan::new(layout, 2, budget, precision).unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        assert_eq!(workspace.precision(), precision);
        assert_eq!(workspace.allocated_bytes(), budget);
        assert!(!workspace.is_poisoned());
        let mut output = op.upload_f32_buffer(&[73.0; 6]).unwrap();
        op.reset_counters();
        op.enable_capture_safe();
        assert!(
            op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut workspace, plan, 2, 3)
                .is_err()
        );
        assert_eq!(op.counters().compute_kernel_launches, 0);
        assert!(!workspace.is_poisoned());
        op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut workspace, plan, 3, 3)
            .unwrap();
        op.disable_capture_safe();
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(
            op.counters().compute_kernel_launches,
            plan.kernel_launches() as u64
        );
        assert!(!workspace.is_poisoned());
        op.record_compute_event().unwrap().synchronize().unwrap();
        assert_eq!(
            op.download_f32_buffer(&output).unwrap(),
            [6.0, 6.0, 6.0, 15.0, 15.0, 15.0]
        );
    }
}
