//! Three frozen linear pairs only: compare reduction independently of RoPE/KV.
use super::*;
use ferrule_backend::cuda::operators::linear::{
    Bf16GemmLayout, CudaNumericFp8Workspace, CudaOperators, F32ToBf16RowsLayout,
};
use ferrule_backend::cuda::providers::DeviceBuffer;
use std::io::Write;

fn words(b: &[u8]) -> Vec<u16> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn emit(file: &mut std::fs::File, value: serde_json::Value) {
    serde_json::to_writer(&mut *file, &value).unwrap();
    writeln!(file).unwrap();
    file.sync_all().unwrap();
}

#[test]
#[ignore = "sole GPU owner; frozen HF shared-down/K/V, decoded operand bytes and local reductions only"]
fn frozen_gqa_kv_and_shared_down_reduction() {
    let (dir, manifest) = fixture();
    gqa_boundary::validate(&manifest).unwrap();
    let decoded_dir = PathBuf::from(
        std::env::var_os("FERRULE_GQA_DECODED_DIR").expect("pinned HF decoded operands"),
    );
    let report = std::env::var_os("FERRULE_GQA_LOCAL_REPORT").expect("fresh JSONL report");
    let mut report = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report)
        .unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut kv_exact = 0;
    for (id, pin) in [
        (
            "l2_SharedExpertDown",
            "ed6f62e58939996c4701568f0fbe53652176cd2cb634cf40607bb6069ef6e179",
        ),
        (
            "l3_AttentionKey",
            "38b73285a39c583c9ce149e5ac51d25ad099e3a57092bfe94999060a889c8760",
        ),
        (
            "l3_AttentionValue",
            "8a490803c4230c18e8a0bb8f0503ead40cb60609c30fbec5e56dbcbc1836c78d",
        ),
    ] {
        let c = manifest.cases.iter().find(|c| c.id == id).unwrap();
        let l = layout(c.n, c.k);
        if matches!(
            c.role,
            TensorRole::AttentionKey | TensorRole::AttentionValue
        ) {
            assert_eq!(
                MODEL
                    .linear_plan(&c.role, l, 1, MODEL_BUDGET)
                    .unwrap()
                    .selected_kernel(),
                K::W8A16FusedBf16Vector
            );
        }
        let decoded_path = decoded_dir.join(format!("{id}.decoded.bf16"));
        let hash = Command::new("sha256sum")
            .arg(&decoded_path)
            .output()
            .unwrap();
        assert!(hash.status.success());
        assert_eq!(
            String::from_utf8(hash.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            pin
        );
        let decoded = std::fs::read(decoded_path).unwrap();
        assert_eq!(decoded.len(), c.n * c.k * 2);
        let input_bytes = bytes(&dir, c, "input.bf16", c.k * 2);
        let input = op.upload_bf16_words(&words(&input_bytes)).unwrap();
        let artifact = op
            .upload_numeric_fp8_linear(
                l,
                &bytes(&dir, c, "weight.fp8", c.n * c.k),
                &bytes(&dir, c, "scale.bf16", c.n / 128 * c.k / 128 * 2),
            )
            .unwrap();
        let expected = words(&bytes(&dir, c, "hf.bf16", c.n * 2));
        let mut out = op.zero_f32_buffer(c.n).unwrap();
        let mut rounded = op.zero_bf16_buffer(c.n).unwrap();
        for p in [
            P::Bf16RneF32Accumulate,
            P::Bf16SlicedK2F32Accumulate,
            P::W8A16Bf16F32Accumulate,
        ] {
            let plan = NumericFp8LinearPlan::new(l, 1, MODEL_BUDGET, p).unwrap();
            let root = DeviceBuffer::<u8>::zeroed(
                &op.stream_clone(),
                plan.workspace_requirements().bytes as usize,
            )
            .unwrap();
            let mut ws = CudaNumericFp8Workspace::from_buffer_with_precision(
                root.slice(0, root.len()).unwrap(),
                p,
            );
            op.reset_counters();
            let before = op.allocator_metrics();
            op.numeric_fp8_bf16_linear_into(&artifact, &input, &mut out, &mut ws, plan, c.k, c.n)
                .unwrap();
            let counts = op.counters();
            assert_eq!(counts.device_allocation_attempts, 0);
            assert_eq!(counts.stream_wide_syncs, 0);
            assert_eq!(
                op.allocator_metrics().live_granted_bytes,
                before.live_granted_bytes
            );
            let raw = op.download_f32_buffer(&out).unwrap();
            let mut operand_exact = None;
            let mut native_raw_exact = None;
            if p == P::Bf16RneF32Accumulate {
                assert_eq!(plan.tile_rows(), c.n);
                let staged = root.to_host_vec(&op.stream_clone()).unwrap();
                assert_eq!(&staged[..c.k * 2], input_bytes);
                assert_eq!(&staged[c.k * 2..], decoded);
                operand_exact = Some(true);
                let weight = op.upload_bf16_words(&words(&decoded)).unwrap();
                let mut native = op.zero_f32_buffer(c.n).unwrap();
                op.bf16_linear_into(
                    &weight,
                    &input,
                    &mut native,
                    Bf16GemmLayout::contiguous(1, c.n, c.k),
                )
                .unwrap();
                let native = op.download_f32_buffer(&native).unwrap();
                assert!(
                    raw.iter()
                        .zip(&native)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                native_raw_exact = Some(true);
            }
            op.f32_rows_to_bf16_rne_into(
                &out,
                &mut rounded,
                F32ToBf16RowsLayout::packed(1, 1, c.n).unwrap(),
            )
            .unwrap();
            let actual = op.download_bf16_buffer(&rounded).unwrap();
            let differences:Vec<_>=actual.iter().zip(&expected).enumerate().filter_map(|(i,(&a,&e))| {
                assert!(a.is_finite()&&raw[i].is_finite());
                let bits=(a.to_bits()>>16) as u16;
                (bits != e).then(||serde_json::json!({"index":i,"actual_bits":format!("0x{bits:04x}"),"hf_bits":format!("0x{e:04x}"),"actual":a,"hf":f32::from_bits(u32::from(e)<<16),"f32_accumulator_bits":format!("0x{:08x}",raw[i].to_bits())}))
            }).collect();
            emit(
                &mut report,
                serde_json::json!({"case":id,"shape":[1,c.n,c.k],"kernel":plan.selected_kernel().name(),"model_selection":MODEL.linear_plan(&c.role,l,1,MODEL_BUDGET).unwrap().selected_kernel().name(),"unequal":differences.len(),"differences":differences,"staged_operands_equal_hf":operand_exact,"native_tiled_f32_equal_numeric":native_raw_exact,"workspace_bytes":plan.workspace_requirements().bytes,"resources_ok":true}),
            );
            if matches!(
                c.role,
                TensorRole::AttentionKey | TensorRole::AttentionValue
            ) && p == P::W8A16Bf16F32Accumulate
            {
                assert!(
                    differences.is_empty(),
                    "candidate must exactly match frozen independent HF"
                );
                kv_exact += 1;
            }
            assert!(!ws.is_poisoned());
        }
    }
    assert_eq!(kv_exact, 2);
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_granted_bytes, 0);
    emit(
        &mut report,
        serde_json::json!({"event":"complete","comparisons":9,"kv_fused_exact":kv_exact,"live_granted_bytes":0,"shared_down_policy":"unchanged; diagnostic only"}),
    );
}
