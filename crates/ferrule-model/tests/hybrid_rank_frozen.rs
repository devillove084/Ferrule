//! Candidate TP2 row-reduction numerical gate, NOT a Mixed decoder acceptance
//! test. Two resident CUDA owners, frozen independent HF inputs/output, no CPU
//! GEMM/reduction. Does not exercise transport, KV/GDN, routing or full40.
#![cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::linear::{CudaOperators, F32ToBf16RowsLayout};
use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use ferrule_model::{
    TensorRole, checkpoint::*, nn::*, transformer::hybrid_parallel::*, transformer::*,
};
use std::{io::Write, path::PathBuf, process::Command, sync::Arc};

fn words(bytes: &[u8]) -> Vec<u16> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| u16::from_le_bytes(*w))
        .collect()
}

#[test]
#[ignore = "explicit two-GPU grant and pinned FERRULE_FROZEN_BF16_DIR required; candidate projection only"]
fn frozen_tp2_bf16_row_partials_reduce_f32_then_round_once() {
    assert_eq!(
        std::env::var("FERRULE_RUN_FROZEN_RANK_GPU").as_deref(),
        Ok("1"),
        "separate explicit GPU grant required"
    );
    let dir = PathBuf::from(
        std::env::var_os("FERRULE_FROZEN_BF16_DIR").expect("frozen fixture required"),
    );
    // Pin the existing source-quality reference, never generate TP expectations.
    let hash = Command::new("sha256sum")
        .arg(dir.join("SHA256SUMS"))
        .output()
        .unwrap();
    assert!(hash.status.success());
    assert_eq!(
        String::from_utf8(hash.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        "59684d1793711f31dd168c5a9154f93a1cddebfeb292ba11c6dcdead62b5bff8"
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "ferrule.frozen-bf16-projections.v1");
    assert_eq!(manifest["oracle"], "independent-pinned-hf-bf16");
    let case = "l0_AttentionOutput";
    let (n, k) = (2048, 4096);
    let input = words(&std::fs::read(dir.join(format!("{case}.input.bf16"))).unwrap());
    let expected = words(&std::fs::read(dir.join(format!("{case}.hf.bf16"))).unwrap());
    assert_eq!(input.len(), k);
    assert_eq!(expected.len(), n);
    let path = ModulePath::new("frozen.output").unwrap();
    let parameter = ParameterSpec::new(
        ParameterId::new(1),
        path.clone(),
        ParameterDType::F8E4M3,
        [n, k],
        ParameterResidency::Layer { layer: 0 },
    )
    .unwrap()
    .with_required_scale(ParameterDType::Bf16, [n / 128, k / 128])
    .unwrap();
    let mut schema = StateDictSchema::builder();
    schema
        .register_with_role(parameter, TensorRole::AttentionOutput)
        .unwrap();
    let mut mapper = ExactNameMapper::new();
    mapper
        .insert("weight", NameMapping::weight(path.clone()))
        .unwrap();
    mapper
        .insert("scale", NameMapping::scale(path.clone()))
        .unwrap();
    let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(vec![
            CheckpointTensorSlice {
                name: "weight".into(),
                role: TensorRole::AttentionOutput,
                path: dir.join(format!("{case}.weight.fp8")),
                offset: 0,
                bytes: (n * k) as u64,
                shape: vec![n, k],
                dtype: CheckpointDType::F8E4M3,
            },
            CheckpointTensorSlice {
                name: "scale".into(),
                role: TensorRole::AttentionOutput,
                path: dir.join(format!("{case}.scale.bf16")),
                offset: 0,
                bytes: (n / 128 * k / 128 * 2) as u64,
                shape: vec![n / 128, k / 128],
                dtype: CheckpointDType::Bf16,
            },
        ])
        .unwrap();
    let binding = bound.get(&path).unwrap();
    use ferrule_model::models::qwen35::{Qwen35Config, Qwen35Recipe};
    let json = serde_json::from_str(include_str!("qwen35_35b_fp8_config.json")).unwrap();
    let spec = Qwen35Recipe::spec(&Qwen35Config::from_value(&json).unwrap()).unwrap();
    let parallel = HybridParallelPlan::new_with_execution(
        &spec,
        0..40,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        vec![ParallelRankId::new(7), ParallelRankId::new(19)],
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    assert!(parallel.validate_rank_execution().is_err());
    let prepared = (0..2)
        .map(|lane| {
            let plan =
                NumericFp8RankPlan::new(binding, Arc::new(parallel.rank_layout(lane).unwrap()))
                    .unwrap();
            StateDictMaterializer::new(plan.budget().payload_bytes as u64)
                .unwrap()
                .prepare_numeric_fp8_rank_linear(&plan, plan.budget().allocation_grant_bytes)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let report_path =
        std::env::var_os("FERRULE_FROZEN_RANK_REPORT").expect("fresh JSON report path required");
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_path)
        .unwrap();
    // Both owners and both paired encoded shards remain live together. No
    // same-device TP simulation and no full-matrix upload on either owner.
    let ops = [
        CudaOperators::new_on_device(0).unwrap(),
        CudaOperators::new_on_device(1).unwrap(),
    ];
    let artifacts = prepared
        .iter()
        .zip(&ops)
        .map(|(p, op)| {
            op.upload_validated_numeric_fp8_linear(p.segments()[0].artifact().validated_payload())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut partials = Vec::new();
    let mut kernels = Vec::new();
    for lane in 0..2 {
        let partition = prepared[lane].plan().partition();
        let columns = partition.columns.clone();
        let layout = NumericFp8Layout {
            n,
            k: columns.len(),
            row_origin: 0,
            column_origin: columns.start,
            scale_type: NumericFp8ScaleType::Bf16,
        };
        // Explicit F32-partial branch. BF16-output GemmEx before the reduction
        // would round twice and is not this proposed reduction contract.
        let plan = NumericFp8Precision::Bf16ModelMixedF32Accumulate
            .linear_plan(&TensorRole::AttentionOutput, layout, 1, 8 << 20)
            .unwrap();
        let op = &ops[lane];
        op.prepare_numeric_fp8_provider(plan, plan.provider_bytes())
            .unwrap();
        let input = op.upload_bf16_words(&input[columns.clone()]).unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        let mut output = op.zero_f32_buffer(n).unwrap();
        op.reset_counters();
        op.numeric_fp8_bf16_linear_into(
            &artifacts[lane],
            &input,
            &mut output,
            &mut workspace,
            plan,
            columns.len(),
            n,
        )
        .unwrap();
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert!(!workspace.is_poisoned());
        kernels.push(format!("{:?}", plan.selected_kernel()));
        // Only bounded communication staging on the host; no host reduction.
        partials.push(op.download_f32_buffer(&output).unwrap());
    }
    let a = ops[0].upload_f32_buffer(&partials[0]).unwrap();
    let b = ops[0].upload_f32_buffer(&partials[1]).unwrap();
    let mut sum = ops[0].zero_f32_buffer(n).unwrap();
    ops[0].residual_add_f32_into(&a, &b, &mut sum).unwrap();
    let mut rounded = ops[0].zero_bf16_buffer(n).unwrap();
    ops[0]
        .f32_rows_to_bf16_rne_into(
            &sum,
            &mut rounded,
            F32ToBf16RowsLayout::packed(1, 1, n).unwrap(),
        )
        .unwrap();
    let actual = ops[0].download_bf16_buffer(&rounded).unwrap();
    let unequal = actual
        .iter()
        .zip(&expected)
        .filter(|(a, e)| a.to_bits() != u32::from(**e) << 16)
        .count();
    let nonfinite = actual.iter().filter(|a| !a.is_finite()).count();
    serde_json::to_writer(
        &mut report,
        &serde_json::json!({"case":case,"tp":2,"ep":1,
        "candidate":"bf16-operands-f32-partials-sum-then-bf16", "kernels":kernels,
        "unequal_bf16_words":unequal,"nonfinite":nonfinite,"model_acceptance":false,
        "reference":"unchanged independent pinned HF BF16", "transport_tested":false}),
    )
    .unwrap();
    writeln!(report).unwrap();
    report.sync_all().unwrap();
    // Strict independent compatibility gate. Do not loosen the current model
    // budget to make TP pass; a failure must inform the chosen TP arithmetic.
    assert_eq!(
        (unequal, nonfinite),
        (0, 0),
        "candidate rank arithmetic differs; inspect frozen report, not model-ready"
    );
}
