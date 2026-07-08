//! Routed MoE step orchestration.
//!
//! This module wires together pieces that must remain independently swappable:
//! router policy, expert residency/streaming planner, artifact reader, expert
//! executor, and optional shared FFN. The first implementation is a CPU reference
//! path for tiny fixtures; CUDA will replace the executor/handle side while
//! keeping the same routing and residency semantics.

use ferrule_common::{Error, Result};

use crate::artifact::binding::RouterArtifactPayload;
use crate::ffn::SwiGluFfnPayload;
use crate::moe::executor::ExpertExecutor;
use crate::moe::handle::{CpuExpertHandleStore, ExpertHandleStore};
use crate::moe::routing::{ExpertRoute, ExpertRouterPolicy};
use crate::moe::streaming::{
    ExpertId, ExpertStreamingPlanner, ExpertStreamingReader, ExpertStreamingStep,
};

#[derive(Debug, Clone, PartialEq)]
pub struct RoutedMoeStepOutput {
    pub routes: Vec<ExpertRoute>,
    pub streaming: ExpertStreamingStep,
    pub routed_output: Vec<f32>,
    pub shared_output: Option<Vec<f32>>,
    pub output: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn execute_routed_moe_reference(
    layer: usize,
    input: &[f32],
    router_logits: &[f32],
    router_bias: Option<&[f32]>,
    hash_experts: Option<&[usize]>,
    predicted_experts: &[usize],
    router_policy: &ExpertRouterPolicy,
    planner: &mut ExpertStreamingPlanner,
    reader: &ExpertStreamingReader,
    expert_executor: &impl ExpertExecutor,
    shared_expert: Option<&SwiGluFfnPayload>,
) -> Result<RoutedMoeStepOutput> {
    let mut handles = CpuExpertHandleStore::new();
    execute_routed_moe_reference_with_handles(
        layer,
        input,
        router_logits,
        router_bias,
        hash_experts,
        predicted_experts,
        router_policy,
        planner,
        reader,
        &mut handles,
        expert_executor,
        shared_expert,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_routed_moe_reference_with_handles(
    layer: usize,
    input: &[f32],
    router_logits: &[f32],
    router_bias: Option<&[f32]>,
    hash_experts: Option<&[usize]>,
    predicted_experts: &[usize],
    router_policy: &ExpertRouterPolicy,
    planner: &mut ExpertStreamingPlanner,
    reader: &ExpertStreamingReader,
    handles: &mut impl ExpertHandleStore,
    expert_executor: &impl ExpertExecutor,
    shared_expert: Option<&SwiGluFfnPayload>,
) -> Result<RoutedMoeStepOutput> {
    let routes = router_policy.route(router_logits, router_bias, hash_experts)?;
    let selected = routes.iter().map(|route| route.expert).collect::<Vec<_>>();
    let streaming = planner.plan_layer_step(layer, &selected, predicted_experts)?;

    handles.apply_evictions(&streaming.evictions);
    for load in &streaming.loads {
        let payload = reader.read_load_source(load.expert, &load.load_source)?;
        handles.insert_artifact_payload(payload)?;
    }

    let mut routed_output = None::<Vec<f32>>;
    for route in &routes {
        let expert_id = ExpertId::new(layer, route.expert);
        let bundle = handles.artifact_bundle(expert_id)?;
        let expert_out = expert_executor.execute(bundle, input, route.weight)?;
        accumulate(&mut routed_output, expert_out)?;
    }
    let routed_output = routed_output.unwrap_or_else(|| vec![0.0; input.len()]);
    let shared_output = shared_expert
        .map(|shared| shared.reference_execute(input, 1.0))
        .transpose()?;
    let mut output = routed_output.clone();
    if let Some(shared) = &shared_output {
        if shared.len() != output.len() {
            return Err(Error::Model(format!(
                "shared expert output length mismatch: routed={}, shared={}",
                output.len(),
                shared.len()
            )));
        }
        for (dst, value) in output.iter_mut().zip(shared.iter()) {
            *dst += value;
        }
    }
    planner.commit_step(&streaming)?;

    Ok(RoutedMoeStepOutput {
        routes,
        streaming,
        routed_output,
        shared_output,
        output,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn execute_routed_moe_with_artifact_router_reference(
    layer: usize,
    input: &[f32],
    token_id: u32,
    router: &RouterArtifactPayload,
    predicted_experts: &[usize],
    router_policy: &ExpertRouterPolicy,
    planner: &mut ExpertStreamingPlanner,
    reader: &ExpertStreamingReader,
    expert_executor: &impl ExpertExecutor,
    shared_expert: Option<&SwiGluFfnPayload>,
) -> Result<RoutedMoeStepOutput> {
    let mut handles = CpuExpertHandleStore::new();
    execute_routed_moe_with_artifact_router_reference_with_handles(
        layer,
        input,
        token_id,
        router,
        predicted_experts,
        router_policy,
        planner,
        reader,
        &mut handles,
        expert_executor,
        shared_expert,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_routed_moe_with_artifact_router_reference_with_handles(
    layer: usize,
    input: &[f32],
    token_id: u32,
    router: &RouterArtifactPayload,
    predicted_experts: &[usize],
    router_policy: &ExpertRouterPolicy,
    planner: &mut ExpertStreamingPlanner,
    reader: &ExpertStreamingReader,
    handles: &mut impl ExpertHandleStore,
    expert_executor: &impl ExpertExecutor,
    shared_expert: Option<&SwiGluFfnPayload>,
) -> Result<RoutedMoeStepOutput> {
    let logits = router.logits(input)?;
    let hash_experts = router.hash_experts_for_token(token_id)?;
    execute_routed_moe_reference_with_handles(
        layer,
        input,
        &logits,
        router.bias.as_deref(),
        hash_experts.as_deref(),
        predicted_experts,
        router_policy,
        planner,
        reader,
        handles,
        expert_executor,
        shared_expert,
    )
}

fn accumulate(target: &mut Option<Vec<f32>>, value: Vec<f32>) -> Result<()> {
    if let Some(target) = target {
        if target.len() != value.len() {
            return Err(Error::Model(format!(
                "routed expert output length mismatch: accumulated={}, next={}",
                target.len(),
                value.len()
            )));
        }
        for (dst, value) in target.iter_mut().zip(value) {
            *dst += value;
        }
    } else {
        *target = Some(value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::TensorRole;

    use super::*;
    use crate::artifact::linear::ArtifactLinearPayload;
    use crate::artifact::tensor::{ArtifactDType, ArtifactTensorPayload, ArtifactTensorSlice};
    use crate::ffn::SwiGluFfnPayload;
    use crate::moe::executor::CpuReferenceExpertExecutor;
    use crate::moe::handle::{CpuExpertHandleStore, ExpertHandleStore};
    use crate::moe::streaming::{
        ExpertArtifactPayload, ExpertLoadSource, ExpertMatrixKind, ExpertStorageTier,
        ExpertStreamingPolicy, ExpertTensorComponent, ExpertTensorKey, ExpertTensorPayload,
        ExpertTensorSlice,
    };

    #[test]
    fn routed_moe_reference_streams_selected_experts_and_adds_shared_ffn() {
        let dir = unique_temp_dir("ferrule-routed-moe-reference");
        std::fs::create_dir_all(&dir).unwrap();
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(2));
        register_tiny_expert(&dir, &mut planner, 0, 0, 0x42, 0x43, 0x22);
        register_tiny_expert(&dir, &mut planner, 0, 1, 0x52, 0x42, 0x22);

        let shared = tiny_shared_ffn();
        let policy = ExpertRouterPolicy::sqrt_softplus_score_topk(2, 1.0);
        let reader = ExpertStreamingReader::new(4096);
        let out = execute_routed_moe_reference(
            0,
            &tiny_input(),
            &[2.0, 1.0],
            None,
            None,
            &[],
            &policy,
            &mut planner,
            &reader,
            &CpuReferenceExpertExecutor::default(),
            Some(&shared),
        )
        .unwrap();

        assert_eq!(
            out.routes
                .iter()
                .map(|route| route.expert)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(out.streaming.loads.len(), 2);
        assert_eq!(out.streaming.evictions.len(), 0);
        assert_eq!(out.output.len(), 32);
        assert!(out.routed_output[0] > 0.0);
        assert!(out.shared_output.as_ref().unwrap()[0] > 0.0);
        assert!(
            (out.output[0] - out.routed_output[0] - out.shared_output.as_ref().unwrap()[0]).abs()
                < 1e-6
        );
        assert!(out.output[1..].iter().all(|value| value.abs() < 1e-6));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tiny_hash_assisted_moe_layer_fixture_uses_artifact_router_streaming_and_shared() {
        let dir = unique_temp_dir("ferrule-tiny-hash-moe-layer");
        std::fs::create_dir_all(&dir).unwrap();
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(2));
        register_tiny_expert(&dir, &mut planner, 0, 0, 0x42, 0x43, 0x22);
        register_tiny_expert(&dir, &mut planner, 0, 1, 0x52, 0x42, 0x22);

        let router = RouterArtifactPayload {
            layer: 0,
            weight: f32_linear(TensorRole::RouterLogits, "router", 2, 32, 0, 1.0),
            bias: None,
            hash_table: Some(vec![1, 0]),
            hash_rows: 1,
            hash_cols: 2,
        };
        let out = execute_routed_moe_with_artifact_router_reference(
            0,
            &tiny_input(),
            0,
            &router,
            &[],
            &ExpertRouterPolicy::sqrt_softplus_hash(2, 1.0),
            &mut planner,
            &ExpertStreamingReader::new(4096),
            &CpuReferenceExpertExecutor::default(),
            Some(&tiny_shared_ffn()),
        )
        .unwrap();
        assert_eq!(
            out.routes
                .iter()
                .map(|route| route.expert)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
        assert_eq!(out.streaming.loads.len(), 2);
        assert!(out.output[0] > 0.0);
        assert!(out.output[1..].iter().all(|value| value.abs() < 1e-6));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn routed_moe_reference_executes_preloaded_resident_expert_handle() {
        let dir = unique_temp_dir("ferrule-routed-moe-resident-handle");
        std::fs::create_dir_all(&dir).unwrap();
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(1));
        register_tiny_expert(&dir, &mut planner, 0, 0, 0x42, 0x43, 0x22);
        planner
            .mark_resident(ExpertId::new(0, 0), ExpertStorageTier::Gpu)
            .unwrap();

        let expert = ExpertId::new(0, 0);
        let mut handles = CpuExpertHandleStore::new();
        handles
            .insert_artifact_payload(ExpertArtifactPayload {
                expert,
                tensors: tiny_expert_tensors(expert, &dir.join("resident.bin"), 0x42, 0x43, 0x22),
            })
            .unwrap();

        let out = execute_routed_moe_reference_with_handles(
            0,
            &tiny_input(),
            &[1.0],
            None,
            None,
            &[],
            &ExpertRouterPolicy::sqrt_softplus_score_topk(1, 1.0),
            &mut planner,
            &ExpertStreamingReader::new(4096),
            &mut handles,
            &CpuReferenceExpertExecutor::default(),
            None,
        )
        .unwrap();
        assert_eq!(out.streaming.loads.len(), 0);
        assert!(out.output[0] > 0.0);
        assert!(handles.contains(expert));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn routed_moe_reference_loads_selected_once_then_uses_cached_handle() {
        let dir = unique_temp_dir("ferrule-routed-moe-cached-handle");
        std::fs::create_dir_all(&dir).unwrap();
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 1,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        register_tiny_expert(&dir, &mut planner, 0, 0, 0x42, 0x43, 0x22);
        let mut handles = CpuExpertHandleStore::new();

        let first = execute_routed_moe_reference_with_handles(
            0,
            &tiny_input(),
            &[1.0],
            None,
            None,
            &[],
            &ExpertRouterPolicy::sqrt_softplus_score_topk(1, 1.0),
            &mut planner,
            &ExpertStreamingReader::new(4096),
            &mut handles,
            &CpuReferenceExpertExecutor::default(),
            None,
        )
        .unwrap();
        assert_eq!(first.streaming.loads.len(), 1);
        assert_eq!(
            planner.location(ExpertId::new(0, 0)),
            Some(ExpertStorageTier::Gpu)
        );
        assert!(handles.contains(ExpertId::new(0, 0)));

        let second = execute_routed_moe_reference_with_handles(
            0,
            &tiny_input(),
            &[1.0],
            None,
            None,
            &[],
            &ExpertRouterPolicy::sqrt_softplus_score_topk(1, 1.0),
            &mut planner,
            &ExpertStreamingReader::new(4096),
            &mut handles,
            &CpuReferenceExpertExecutor::default(),
            None,
        )
        .unwrap();
        assert_eq!(second.streaming.loads.len(), 0);
        assert_eq!(first.output, second.output);

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn register_tiny_expert(
        dir: &Path,
        planner: &mut ExpertStreamingPlanner,
        layer: usize,
        expert: usize,
        gate_byte: u8,
        up_byte: u8,
        down_byte: u8,
    ) {
        let expert_id = ExpertId::new(layer, expert);
        let path = dir.join(format!("l{layer}e{expert}.bin"));
        let tensors = tiny_expert_tensors(expert_id, &path, gate_byte, up_byte, down_byte);
        let mut bytes = Vec::new();
        for tensor in &tensors {
            bytes.extend(&tensor.bytes);
        }
        std::fs::write(&path, bytes).unwrap();
        let mut offset = 0u64;
        let slices = tensors
            .into_iter()
            .map(|tensor| {
                let bytes = tensor.bytes.len() as u64;
                let slice = ExpertTensorSlice {
                    offset,
                    bytes,
                    ..tensor.slice
                };
                offset += bytes;
                slice
            })
            .collect();
        planner.register_load_source(
            expert_id,
            ExpertLoadSource::LocalTensorSet { tensors: slices },
        );
    }

    fn tiny_expert_tensors(
        expert: ExpertId,
        path: &Path,
        gate_byte: u8,
        up_byte: u8,
        down_byte: u8,
    ) -> Vec<ExpertTensorPayload> {
        vec![
            tiny_fp4_payload(
                expert,
                path,
                ExpertMatrixKind::Gate,
                ExpertTensorComponent::Weight,
                gate_byte,
            ),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Gate),
            tiny_fp4_payload(
                expert,
                path,
                ExpertMatrixKind::Up,
                ExpertTensorComponent::Weight,
                up_byte,
            ),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Up),
            tiny_fp4_payload(
                expert,
                path,
                ExpertMatrixKind::Down,
                ExpertTensorComponent::Weight,
                down_byte,
            ),
            tiny_scale_payload(expert, path, ExpertMatrixKind::Down),
        ]
    }

    fn tiny_fp4_payload(
        expert: ExpertId,
        path: &Path,
        matrix: ExpertMatrixKind,
        component: ExpertTensorComponent,
        first_byte: u8,
    ) -> ExpertTensorPayload {
        let mut bytes = vec![0u8; 32 * 16];
        bytes[0] = first_byte;
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component,
                path: path.to_path_buf(),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: "I8".into(),
                shape: vec![32, 16],
            },
            bytes,
        }
    }

    fn tiny_scale_payload(
        expert: ExpertId,
        path: &Path,
        matrix: ExpertMatrixKind,
    ) -> ExpertTensorPayload {
        let bytes = vec![127u8; 32];
        ExpertTensorPayload {
            slice: ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Scale,
                path: path.to_path_buf(),
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: "F8_E8M0".into(),
                shape: vec![32, 1],
            },
            bytes,
        }
    }

    fn tiny_shared_ffn() -> SwiGluFfnPayload {
        SwiGluFfnPayload {
            gate: f32_linear(TensorRole::SharedExpertGate, "shared_gate", 1, 32, 0, 1.0),
            up: f32_linear(TensorRole::SharedExpertUp, "shared_up", 1, 32, 1, 1.0),
            down: f32_linear(TensorRole::SharedExpertDown, "shared_down", 32, 1, 0, 1.0),
            swiglu_limit: 0.0,
        }
    }

    fn f32_linear(
        role: TensorRole,
        name: &str,
        out: usize,
        input: usize,
        nonzero_col: usize,
        value: f32,
    ) -> ArtifactLinearPayload {
        let mut values = vec![0.0f32; out * input];
        values[nonzero_col] = value;
        ArtifactLinearPayload::from_weight_and_scale(
            role,
            ArtifactTensorPayload {
                slice: ArtifactTensorSlice {
                    name: format!("{name}.weight"),
                    role: TensorRole::Unknown,
                    path: PathBuf::from("synthetic.safetensors"),
                    offset: 0,
                    bytes: (values.len() * 4) as u64,
                    dtype: ArtifactDType::F32,
                    shape: vec![out, input],
                },
                bytes: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            },
            None,
        )
        .unwrap()
    }

    fn tiny_input() -> Vec<f32> {
        let mut input = vec![0.0f32; 32];
        input[0] = 2.0;
        input[1] = 3.0;
        input
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
