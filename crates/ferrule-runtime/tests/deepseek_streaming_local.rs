use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};

use ferrule_model::{
    ArtifactGroupKind, ArtifactLinearFormat, ArtifactLinearPayload, ArtifactTensorReader,
    ArtifactTensorSlice, ExpertComputeBundle, ExpertId, ExpertLinearFormat, ExpertLoadReason,
    ExpertLoadSource, ExpertRouterPolicy, ExpertStorageTier, ExpertStreamingPlanner,
    ExpertStreamingPolicy, ExpertStreamingReader, HfSafetensorsInventory, HfSafetensorsTensorInfo,
    HyperConnectionConfig, ModelDescriptor, ModelFamily, SparseAttentionSpec, TensorRole,
    TokenizerHandle, bind_attention_from_artifact_group, bind_attention_from_hf,
    bind_hyper_connection_from_artifact_group, bind_hyper_connection_from_hf,
    bind_hyper_connection_head_from_hf, bind_layer_norms_from_artifact_group,
    bind_router_from_artifact_group, bind_router_from_hf,
    bind_shared_swiglu_ffn_from_artifact_group, bind_shared_swiglu_ffn_from_hf,
    families::deepseek_v4,
    models::deepseek_v4::{DeepSeekV4ArtifactModel, DeepSeekV4PrepareOptions, DeepSeekV4Runner},
    semantic::{AttentionTensorKind, HyperConnectionStage, RouterTensorKind},
};
use ferrule_runtime::{
    BackendObject, bind_layer_artifact_from_hf, build_graph_program_from_descriptor,
    materialize_graph_hf_externals, validate_graph_program,
};

#[test]
fn local_deepseek_v4_expert_streaming_reads_one_selected_expert_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let routed = inventory.routed_expert_tensors();
    assert_eq!(
        routed.len(),
        deepseek_v4::NUM_LAYERS * deepseek_v4::N_ROUTED_EXPERTS * 3 * 2
    );

    let unique_experts = routed
        .iter()
        .map(|tensor| (tensor.descriptor.layer, tensor.descriptor.expert))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        unique_experts.len(),
        deepseek_v4::NUM_LAYERS * deepseek_v4::N_ROUTED_EXPERTS
    );

    let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(6));
    assert_eq!(
        planner
            .register_hf_routed_expert_tensor_sets(&model_dir, routed)
            .expect("expert tensor sets should register"),
        deepseek_v4::NUM_LAYERS * deepseek_v4::N_ROUTED_EXPERTS
    );

    let step = planner
        .plan_layer_step(0, &[0, 1, 2, 3, 4, 5], &[])
        .expect("selected experts should be plannable");
    assert_eq!(step.loads.len(), 6);
    assert!(
        step.loads
            .iter()
            .all(|load| load.reason == ExpertLoadReason::Selected)
    );

    let load = step
        .loads
        .iter()
        .find(|load| load.expert == ExpertId::new(0, 0))
        .expect("layer 0 expert 0 should be selected");
    let ExpertLoadSource::LocalTensorSet { tensors } = &load.load_source else {
        panic!("expected LocalTensorSet for HF expert artifact");
    };
    assert_eq!(tensors.len(), 6);
    assert!(tensors.iter().all(|tensor| tensor.path.exists()));
    assert!(tensors.iter().all(|tensor| tensor.bytes > 0));

    let reader = ExpertStreamingReader::new(64 * 1024 * 1024);
    let payload = reader
        .read_load_source(load.expert, &load.load_source)
        .expect("bounded reader should read one selected expert from local shards");
    assert_eq!(payload.expert, ExpertId::new(0, 0));
    assert_eq!(payload.tensors.len(), 6);
    assert_eq!(
        payload
            .tensors
            .iter()
            .map(|tensor| tensor.bytes.len() as u64)
            .sum::<u64>(),
        load.load_source.bytes()
    );
    assert!(
        payload
            .tensors
            .iter()
            .any(|tensor| tensor.slice.dtype == "I8")
    );
    assert!(
        payload
            .tensors
            .iter()
            .any(|tensor| tensor.slice.dtype == "F8_E8M0")
    );
    let bundle = ExpertComputeBundle::from_artifact_payload(payload)
        .expect("artifact slices should form a FP4 expert compute bundle");
    assert_eq!(
        bundle.gate.format,
        ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features: deepseek_v4::MOE_INTERMEDIATE_SIZE,
            in_features: deepseek_v4::HIDDEN_SIZE,
            block_size: 32,
        }
    );
    assert_eq!(
        bundle.down.format,
        ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features: deepseek_v4::HIDDEN_SIZE,
            in_features: deepseek_v4::MOE_INTERMEDIATE_SIZE,
            block_size: 32,
        }
    );
}

#[test]
fn local_deepseek_v4_shared_expert_binds_real_layer0_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let shared = inventory.shared_expert_tensors();
    assert_eq!(shared.len(), deepseek_v4::NUM_LAYERS * 3 * 2);

    let ffn = bind_shared_swiglu_ffn_from_hf(
        &model_dir,
        0,
        &shared,
        &ArtifactTensorReader::new(64 * 1024 * 1024),
        deepseek_v4::SWIGLU_LIMIT,
    )
    .expect("layer 0 shared expert should bind into artifact linears");
    assert_eq!(ffn.gate.format.in_features(), deepseek_v4::HIDDEN_SIZE);
    assert_eq!(
        ffn.gate.format.out_features(),
        deepseek_v4::MOE_INTERMEDIATE_SIZE
    );
    assert_eq!(ffn.up.format.in_features(), deepseek_v4::HIDDEN_SIZE);
    assert_eq!(
        ffn.up.format.out_features(),
        deepseek_v4::MOE_INTERMEDIATE_SIZE
    );
    assert_eq!(
        ffn.down.format.in_features(),
        deepseek_v4::MOE_INTERMEDIATE_SIZE
    );
    assert_eq!(ffn.down.format.out_features(), deepseek_v4::HIDDEN_SIZE);
}

#[test]
fn local_deepseek_v4_router_binds_hash_and_score_layers_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let routers = inventory.router_tensors();
    assert_eq!(
        routers
            .iter()
            .filter(|tensor| tensor.descriptor.kind == RouterTensorKind::Weight)
            .count(),
        deepseek_v4::NUM_LAYERS
    );
    assert_eq!(
        routers
            .iter()
            .filter(|tensor| tensor.descriptor.kind == RouterTensorKind::HashTable)
            .count(),
        deepseek_v4::NUM_HASH_LAYERS
    );
    assert_eq!(
        routers
            .iter()
            .filter(|tensor| tensor.descriptor.kind == RouterTensorKind::Bias)
            .count(),
        deepseek_v4::NUM_LAYERS - deepseek_v4::NUM_HASH_LAYERS
    );

    let reader = ArtifactTensorReader::new(64 * 1024 * 1024);
    let hash_router = bind_router_from_hf(&model_dir, 0, &routers, &reader)
        .expect("layer 0 hash router should bind");
    assert_eq!(
        hash_router.weight.format.out_features(),
        deepseek_v4::N_ROUTED_EXPERTS
    );
    assert_eq!(
        hash_router.weight.format.in_features(),
        deepseek_v4::HIDDEN_SIZE
    );
    assert!(hash_router.bias.is_none());
    let hash = hash_router
        .hash_experts_for_token(0)
        .expect("token 0 should be in hash table")
        .expect("layer 0 should have a hash table");
    assert_eq!(hash.len(), deepseek_v4::NUM_EXPERTS_PER_TOK);
    assert!(
        hash.iter()
            .all(|expert| *expert < deepseek_v4::N_ROUTED_EXPERTS)
    );

    let score_router = bind_router_from_hf(&model_dir, 3, &routers, &reader)
        .expect("layer 3 score router should bind");
    assert_eq!(
        score_router.weight.format.out_features(),
        deepseek_v4::N_ROUTED_EXPERTS
    );
    assert_eq!(
        score_router.weight.format.in_features(),
        deepseek_v4::HIDDEN_SIZE
    );
    assert_eq!(
        score_router.bias.as_ref().map(Vec::len),
        Some(deepseek_v4::N_ROUTED_EXPERTS)
    );
    assert!(score_router.hash_table.is_none());
}

#[test]
fn local_deepseek_v4_artifact_linear_reads_attention_fp8_pair_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let weight = inventory_tensor(&inventory, "layers.0.attn.wq_a.weight");
    let scale = inventory_tensor(&inventory, "layers.0.attn.wq_a.scale");
    assert_eq!(weight.dtype, "F8_E4M3");
    assert_eq!(scale.dtype, "F8_E8M0");

    let reader = ArtifactTensorReader::new(64 * 1024 * 1024);
    let weight = reader
        .read_slice(&ArtifactTensorSlice::from_hf_inventory(&model_dir, weight))
        .expect("wq_a weight should be readable as a bounded artifact tensor");
    let scale = reader
        .read_slice(&ArtifactTensorSlice::from_hf_inventory(&model_dir, scale))
        .expect("wq_a scale should be readable as a bounded artifact tensor");
    let linear = ArtifactLinearPayload::from_weight_and_scale(
        TensorRole::AttentionLatentQueryA,
        weight,
        Some(scale),
    )
    .expect("wq_a weight/scale should infer an FP8 artifact linear format");
    assert_eq!(
        linear.format,
        ArtifactLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: deepseek_v4::Q_LORA_RANK,
            in_features: deepseek_v4::HIDDEN_SIZE,
            block_m: 128,
            block_k: 128,
        }
    );
}

#[test]
fn local_deepseek_v4_attention_and_hc_bind_real_artifacts_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let attention = inventory.attention_tensors();
    let core_attention = attention
        .iter()
        .filter(|tensor| {
            !matches!(
                tensor.descriptor.kind,
                AttentionTensorKind::Compressor | AttentionTensorKind::Indexer
            )
        })
        .count();
    let compressor = attention
        .iter()
        .filter(|tensor| tensor.descriptor.kind == AttentionTensorKind::Compressor)
        .count();
    let indexer = attention
        .iter()
        .filter(|tensor| tensor.descriptor.kind == AttentionTensorKind::Indexer)
        .count();
    assert_eq!(core_attention, deepseek_v4::NUM_LAYERS * 13);
    assert_eq!(compressor, 164);
    assert_eq!(indexer, 147);

    let reader = ArtifactTensorReader::new(64 * 1024 * 1024);
    let layer0 = bind_attention_from_hf(&model_dir, 0, &attention, &reader)
        .expect("layer 0 attention artifact payload should bind");
    assert_eq!(
        layer0.query_a.format,
        ArtifactLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: deepseek_v4::Q_LORA_RANK,
            in_features: deepseek_v4::HIDDEN_SIZE,
            block_m: 128,
            block_k: 128,
        }
    );
    assert_eq!(
        layer0.query_b.format,
        ArtifactLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: deepseek_v4::NUM_HEADS * deepseek_v4::HEAD_DIM,
            in_features: deepseek_v4::Q_LORA_RANK,
            block_m: 128,
            block_k: 128,
        }
    );
    assert_eq!(
        layer0.key_value.format,
        ArtifactLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: deepseek_v4::HEAD_DIM,
            in_features: deepseek_v4::HIDDEN_SIZE,
            block_m: 128,
            block_k: 128,
        }
    );
    assert_eq!(layer0.query_norm.len(), deepseek_v4::Q_LORA_RANK);
    assert_eq!(layer0.key_value_norm.len(), deepseek_v4::HEAD_DIM);
    assert_eq!(layer0.attention_sink.len(), deepseek_v4::NUM_HEADS);
    assert!(layer0.auxiliary.is_empty());

    let layer2 = bind_attention_from_hf(&model_dir, 2, &attention, &reader)
        .expect("layer 2 attention artifact payload should bind with compressor/indexer auxiliary");
    assert!(!layer2.auxiliary.is_empty());

    let hc = inventory.hyper_connection_tensors();
    assert_eq!(hc.len(), deepseek_v4::NUM_LAYERS * 6 + 3);
    let config = HyperConnectionConfig {
        hc_mult: deepseek_v4::HC_MULT,
        hidden_size: deepseek_v4::HIDDEN_SIZE,
        sinkhorn_iters: 4,
        eps: deepseek_v4::HC_EPS,
        norm_eps: deepseek_v4::RMS_NORM_EPS,
    };
    let hc_attn = bind_hyper_connection_from_hf(
        &model_dir,
        0,
        HyperConnectionStage::Attention,
        &hc,
        &reader,
        config,
    )
    .expect("layer 0 attention HC should bind");
    assert_eq!(
        hc_attn.function.len(),
        config.mix_hc() * config.hc_hidden_size()
    );
    assert_eq!(hc_attn.scale.len(), 3);
    assert_eq!(hc_attn.base.len(), config.mix_hc());

    let hc_ffn = bind_hyper_connection_from_hf(
        &model_dir,
        0,
        HyperConnectionStage::FeedForward,
        &hc,
        &reader,
        config,
    )
    .expect("layer 0 FFN HC should bind");
    assert_eq!(
        hc_ffn.function.len(),
        config.mix_hc() * config.hc_hidden_size()
    );
    assert_eq!(hc_ffn.scale.len(), 3);
    assert_eq!(hc_ffn.base.len(), config.mix_hc());

    let hc_head = bind_hyper_connection_head_from_hf(&model_dir, &hc, &reader, config)
        .expect("global HC head should bind");
    assert_eq!(
        hc_head.function.len(),
        config.hc_mult * config.hc_hidden_size()
    );
    assert_eq!(hc_head.scale.len(), 1);
    assert_eq!(hc_head.base.len(), config.hc_mult);
}

#[test]
fn local_deepseek_v4_layer_artifact_bundle_binds_real_layer0_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let attention = inventory.attention_tensors();
    let hc = inventory.hyper_connection_tensors();
    let routers = inventory.router_tensors();
    let shared = inventory.shared_expert_tensors();
    let reader = ArtifactTensorReader::new(64 * 1024 * 1024);
    let config = HyperConnectionConfig {
        hc_mult: deepseek_v4::HC_MULT,
        hidden_size: deepseek_v4::HIDDEN_SIZE,
        sinkhorn_iters: 4,
        eps: deepseek_v4::HC_EPS,
        norm_eps: deepseek_v4::RMS_NORM_EPS,
    };
    let binding = bind_layer_artifact_from_hf(
        &model_dir,
        0,
        &attention,
        &hc,
        &routers,
        &shared,
        &reader,
        config,
        deepseek_v4::SWIGLU_LIMIT,
        ExpertRouterPolicy::sqrt_softplus_hash(
            deepseek_v4::NUM_EXPERTS_PER_TOK,
            deepseek_v4::ROUTED_SCALING_FACTOR,
        ),
        SparseAttentionSpec {
            heads: deepseek_v4::NUM_HEADS,
            head_dim: deepseek_v4::HEAD_DIM,
            topk: deepseek_v4::SLIDING_WINDOW,
            softmax_scale: (deepseek_v4::HEAD_DIM as f32).powf(-0.5),
            has_attention_sink: true,
        },
    )
    .expect("layer 0 artifact bundle should compose real attention/HC/router/shared payloads");
    assert_eq!(binding.layer, 0);
    assert_eq!(
        binding.attention.query_a.format.in_features(),
        deepseek_v4::HIDDEN_SIZE
    );
    assert_eq!(
        binding.attention.attention_sink.len(),
        deepseek_v4::NUM_HEADS
    );
    assert!(binding.router.hash_table.is_some());
    assert!(binding.shared_ffn.is_some());
    assert_eq!(binding.hc_attention.scale.len(), 3);
    assert_eq!(binding.hc_feed_forward.scale.len(), 3);
}

#[test]
fn local_deepseek_v4_semantic_graph_materializes_artifact_groups_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let descriptor = ModelDescriptor::load(&model_dir)
        .expect("local DeepSeek V4 descriptor should load from config/index metadata");
    let program = build_graph_program_from_descriptor(&descriptor)
        .expect("semantic graph program should build through generic graph builder");
    validate_graph_program(&program).expect("semantic graph program should validate");
    assert!(
        program.graph.nodes().iter().all(|node| !node
            .op()
            .name()
            .to_ascii_lowercase()
            .contains("deepseek"))
    );
    assert!(program.bindings.entries().iter().all(|binding| {
        let name = binding.key.name();
        !name.contains("layers.0.attn.wq_a") && !name.contains("model.layers.0")
    }));
    assert_eq!(
        program.semantic_plan.layers[0].feed_forward.router,
        ferrule_model::RouterKind::HashAssistedTopK
    );
    assert_eq!(
        program.semantic_plan.layers[deepseek_v4::NUM_HASH_LAYERS]
            .feed_forward
            .router,
        ferrule_model::RouterKind::DenseTopK
    );
    assert_eq!(
        program.semantic_plan.layers[0].feed_forward.swiglu_limit,
        Some(deepseek_v4::SWIGLU_LIMIT)
    );
    assert_eq!(
        program.semantic_plan.layers[0].feed_forward.route_scale,
        Some(deepseek_v4::ROUTED_SCALING_FACTOR)
    );
    assert_eq!(
        program.semantic_plan.layers[0].attention.window_size,
        Some(deepseek_v4::SLIDING_WINDOW)
    );

    let inventory = HfSafetensorsInventory::open(&model_dir, descriptor.spec.family.clone())
        .expect("local DeepSeek V4 inventory should parse headers only");
    let objects = materialize_graph_hf_externals(&program, &inventory, &model_dir)
        .expect("semantic graph externals should materialize to artifact groups");

    let layer0 = objects
        .layer_objects(0)
        .expect("layer 0 graph objects should aggregate from semantic externals");
    assert_eq!(layer0.layer, 0);
    assert_eq!(layer0.attention.kind, ArtifactGroupKind::Attention);
    assert!(!layer0.attention.tensors.is_empty());
    assert!(
        !layer0
            .layer_norms
            .expect("layer 0 stage norm artifacts should aggregate")
            .tensors
            .is_empty()
    );
    assert!(
        !layer0
            .hc_attention
            .expect("layer 0 HC attention artifacts should aggregate")
            .tensors
            .is_empty()
    );
    assert!(
        !layer0
            .hc_feed_forward
            .expect("layer 0 HC FFN artifacts should aggregate")
            .tensors
            .is_empty()
    );
    assert!(
        !layer0
            .router
            .expect("layer 0 router artifacts should aggregate")
            .tensors
            .is_empty()
    );
    assert!(
        !layer0
            .shared_expert
            .expect("layer 0 shared expert artifacts should aggregate")
            .tensors
            .is_empty()
    );
    assert!(layer0.uses_hyper_connection());
    assert!(layer0.uses_routed_experts());
    assert!(layer0.uses_shared_expert());
    assert!(matches!(
        layer0
            .kv_state
            .expect("layer 0 KV state should aggregate")
            .object,
        BackendObject::KvState(None)
    ));

    let registry = layer0
        .expert_registry
        .expect("layer 0 routed experts should materialize as an expert registry");
    assert_eq!(registry.experts.len(), deepseek_v4::N_ROUTED_EXPERTS);

    let reader = ArtifactTensorReader::new(64 * 1024 * 1024);
    let attention = bind_attention_from_artifact_group(layer0.attention, &reader)
        .expect("layer 0 graph attention group should bind into an attention payload");
    assert_eq!(
        attention.query_a.format.in_features(),
        deepseek_v4::HIDDEN_SIZE
    );
    assert_eq!(attention.attention_sink.len(), deepseek_v4::NUM_HEADS);
    let layer_norms = bind_layer_norms_from_artifact_group(
        layer0
            .layer_norms
            .expect("layer 0 stage norm artifacts should aggregate"),
        &reader,
    )
    .expect("layer 0 graph stage norm group should bind into norm payloads");
    assert_eq!(
        layer_norms.attention_norm.as_ref().map(Vec::len),
        Some(deepseek_v4::HIDDEN_SIZE)
    );
    assert_eq!(
        layer_norms.feed_forward_norm.as_ref().map(Vec::len),
        Some(deepseek_v4::HIDDEN_SIZE)
    );

    let config = HyperConnectionConfig {
        hc_mult: deepseek_v4::HC_MULT,
        hidden_size: deepseek_v4::HIDDEN_SIZE,
        sinkhorn_iters: 4,
        eps: deepseek_v4::HC_EPS,
        norm_eps: deepseek_v4::RMS_NORM_EPS,
    };
    let hc_attention = bind_hyper_connection_from_artifact_group(
        layer0
            .hc_attention
            .expect("layer 0 HC attention artifacts should aggregate"),
        &reader,
        config,
    )
    .expect("layer 0 graph HC attention group should bind into HC weights");
    assert_eq!(hc_attention.scale.len(), 3);
    let hc_ffn = bind_hyper_connection_from_artifact_group(
        layer0
            .hc_feed_forward
            .expect("layer 0 HC FFN artifacts should aggregate"),
        &reader,
        config,
    )
    .expect("layer 0 graph HC FFN group should bind into HC weights");
    assert_eq!(hc_ffn.scale.len(), 3);

    let router = bind_router_from_artifact_group(
        layer0
            .router
            .expect("layer 0 router artifacts should aggregate"),
        &reader,
    )
    .expect("layer 0 graph router group should bind into router payload");
    assert_eq!(
        router.weight.format.out_features(),
        deepseek_v4::N_ROUTED_EXPERTS
    );
    assert!(router.hash_table.is_some());

    let shared = bind_shared_swiglu_ffn_from_artifact_group(
        layer0
            .shared_expert
            .expect("layer 0 shared expert artifacts should aggregate"),
        &reader,
        deepseek_v4::SWIGLU_LIMIT,
    )
    .expect("layer 0 graph shared expert group should bind into shared FFN payload");
    assert_eq!(shared.gate.format.in_features(), deepseek_v4::HIDDEN_SIZE);
}

#[test]
fn local_deepseek_v4_model_binds_top_level_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 64 * 1024 * 1024)
        .expect("local DeepSeek V4 top-level artifact model should bind real metadata/HC head");
    assert_eq!(model.embedding.rows, deepseek_v4::VOCAB_SIZE);
    assert_eq!(model.embedding.cols, deepseek_v4::HIDDEN_SIZE);
    assert_eq!(model.output_norm.len(), deepseek_v4::HIDDEN_SIZE);
    assert_eq!(model.output_head.rows, deepseek_v4::VOCAB_SIZE);
    assert_eq!(model.output_head.cols, deepseek_v4::HIDDEN_SIZE);
    assert_eq!(model.hc_head.scale.len(), 1);
    assert_eq!(model.model_info().backend, "deepseek-v4-artifact");
}

#[test]
fn local_deepseek_v4_model_reads_real_embedding_and_output_head_rows_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 64 * 1024 * 1024)
        .expect("local DeepSeek V4 top-level artifact model should bind real metadata/HC head");
    let embedding = model
        .embedding_for_token(0)
        .expect("single embedding row should be readable without loading the full table");
    assert_eq!(embedding.len(), deepseek_v4::HIDDEN_SIZE);
    assert!(embedding.iter().all(|value| value.is_finite()));

    let hc_state = model
        .initial_hc_state_for_token(0)
        .expect("embedding row should expand into HC state");
    assert_eq!(
        hc_state.len(),
        deepseek_v4::HC_MULT * deepseek_v4::HIDDEN_SIZE
    );
    let hidden = model
        .normalized_hidden_from_hc_state(&hc_state)
        .expect("HC head + output norm should run on real top-level tensors");
    assert_eq!(hidden.len(), deepseek_v4::HIDDEN_SIZE);
    let logits = model
        .logits_for_hidden_row_range(&hidden, 0, 8)
        .expect("lm_head rows should be readable and executable in small chunks");
    assert_eq!(logits.len(), 8);
    assert!(logits.iter().all(|value| value.is_finite()));
}

#[test]
fn local_deepseek_v4_runner_decodes_top_level_logits_rows_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let options = DeepSeekV4PrepareOptions {
        max_layers: 0,
        output_head_chunk_rows: 8,
        expert_reader_max_tensor_bytes: 64 * 1024 * 1024,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let mut runner = DeepSeekV4Runner::load_hf_with_options(&model_dir, 64 * 1024 * 1024, options)
        .expect("local DeepSeek V4 runner should load real top-level artifacts");
    let logits = runner
        .decode_token_logits_row_range(0, 0, 8)
        .expect("max_layers=0 runner should produce real top-level logits row range");
    assert_eq!(logits.len(), 8);
    assert_eq!(runner.position(), 1);
    assert!(logits.iter().all(|value| value.is_finite()));
    runner.reset().expect("runner reset should succeed");
    assert_eq!(runner.position(), 0);
}

#[test]
fn local_deepseek_v4_runner_prefills_prompt_top_level_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let options = DeepSeekV4PrepareOptions {
        max_layers: 0,
        output_head_chunk_rows: 8,
        expert_reader_max_tensor_bytes: 64 * 1024 * 1024,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let mut runner = DeepSeekV4Runner::load_hf_with_options(&model_dir, 64 * 1024 * 1024, options)
        .expect("local DeepSeek V4 runner should load real top-level artifacts");
    let token_ids = runner
        .model()
        .tokenizer
        .encode("Ferrule DSV4 prefill smoke")
        .expect("local tokenizer should encode prompt");
    assert!(!token_ids.is_empty());
    let logits = runner
        .prefill_tokens_logits_row_range(&token_ids, 0, 8)
        .expect("sequential prefill fallback should produce row logits");
    assert_eq!(logits.len(), 8);
    assert_eq!(runner.position(), token_ids.len());
    assert_eq!(runner.bound_layer_count(), 0);
    assert!(logits.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "expensive: executes real DSV4 layer-0 CPU reference path over local shards"]
fn local_deepseek_v4_runner_decodes_one_real_layer_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let options = DeepSeekV4PrepareOptions {
        max_layers: 1,
        output_head_chunk_rows: 8,
        expert_reader_max_tensor_bytes: 64 * 1024 * 1024,
        moe_prefetch_experts: 0,
        moe_hotset_experts: 0,
        ..DeepSeekV4PrepareOptions::default()
    };
    let mut runner = DeepSeekV4Runner::load_hf_with_options(&model_dir, 128 * 1024 * 1024, options)
        .expect("local DeepSeek V4 runner should load real artifacts");
    let logits = runner
        .decode_token_logits_row_range(0, 0, 1)
        .expect("layer-0 reference path should produce one logit row");
    assert_eq!(logits.len(), 1);
    assert_eq!(runner.position(), 1);
    assert_eq!(runner.bound_layer_count(), 1);
    assert!(logits[0].is_finite());
}

#[test]
fn local_deepseek_v4_model_binds_layer0_with_official_shapes_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 64 * 1024 * 1024)
        .expect("local DeepSeek V4 artifact model should bind top-level state");
    let layer = model
        .bind_layer(0)
        .expect("local DeepSeek V4 layer0 should bind through model-specific runtime boundary");
    assert_eq!(layer.layer, 0);
    assert_eq!(layer.attn_norm.len(), deepseek_v4::HIDDEN_SIZE);
    assert_eq!(layer.ffn_norm.len(), deepseek_v4::HIDDEN_SIZE);
    assert_eq!(layer.attention.config.compress_ratio, 0);
    assert_eq!(
        layer.attention.config.output_group_input_dim(),
        deepseek_v4::NUM_HEADS * deepseek_v4::HEAD_DIM / deepseek_v4::O_GROUPS
    );
    assert_eq!(
        layer.attention.payload.output_a.format.in_features(),
        deepseek_v4::NUM_HEADS * deepseek_v4::HEAD_DIM / deepseek_v4::O_GROUPS
    );
    assert_eq!(
        layer.attention.payload.output_a.format.out_features(),
        deepseek_v4::O_GROUPS * deepseek_v4::O_LORA_RANK
    );
    assert!(layer.router.hash_table.is_some());
    assert_eq!(layer.hc_attention.scale.len(), 3);
    assert_eq!(layer.hc_feed_forward.scale.len(), 3);
}

#[test]
fn local_deepseek_v4_compressed_attention_payloads_bind_official_shapes_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 64 * 1024 * 1024)
        .expect("local DeepSeek V4 artifact model should bind top-level state");
    let layer2 = model
        .bind_layer(2)
        .expect("local DeepSeek V4 layer2 should bind compressed attention artifacts");
    assert_eq!(layer2.attention.config.compress_ratio, 4);
    let compressed = layer2
        .attention
        .compressed
        .as_ref()
        .expect("layer2 should have typed compressed attention payload");
    assert_eq!(compressed.compressor.ape_rows, 4);
    assert_eq!(compressed.compressor.ape_cols, 2 * deepseek_v4::HEAD_DIM);
    assert_eq!(
        compressed.compressor.wkv.format.out_features(),
        2 * deepseek_v4::HEAD_DIM
    );
    assert_eq!(
        compressed.compressor.wgate.format.in_features(),
        deepseek_v4::HIDDEN_SIZE
    );
    let indexer = compressed
        .indexer
        .as_ref()
        .expect("ratio-4 layers should bind indexer payload");
    assert_eq!(indexer.compressor.head_dim, deepseek_v4::INDEX_HEAD_DIM);
    assert_eq!(
        indexer.wq_b.format.out_features(),
        deepseek_v4::INDEX_N_HEADS * deepseek_v4::INDEX_HEAD_DIM
    );
    assert_eq!(
        indexer.weights_proj.format.out_features(),
        deepseek_v4::INDEX_N_HEADS
    );

    let layer3 = model
        .bind_layer(3)
        .expect("local DeepSeek V4 layer3 should bind ratio-128 compressor artifacts");
    assert_eq!(layer3.attention.config.compress_ratio, 128);
    let compressed = layer3
        .attention
        .compressed
        .as_ref()
        .expect("layer3 should have typed compressor payload");
    assert!(compressed.indexer.is_none());
    assert_eq!(compressed.compressor.ape_rows, 128);
    assert_eq!(compressed.compressor.ape_cols, deepseek_v4::HEAD_DIM);
    assert_eq!(
        compressed.compressor.wkv.format.out_features(),
        deepseek_v4::HEAD_DIM
    );
}

#[test]
fn local_deepseek_v4_layer_state_registers_real_routed_expert_artifacts_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let model = DeepSeekV4ArtifactModel::load_hf_with_limit(&model_dir, 64 * 1024 * 1024)
        .expect("local DeepSeek V4 artifact model should bind top-level state");
    let state = model
        .new_layer_sequence_state(0)
        .expect("local DeepSeek V4 layer sequence state should initialize");
    let expert_runtime = model
        .new_quality_first_layer_expert_runtime_with_residency(0, 0, 0)
        .expect("local DeepSeek V4 expert runtime should register routed artifacts");
    assert_eq!(state.kv.len(), 0);
    assert_eq!(
        expert_runtime.expert_planner.location(ExpertId::new(0, 0)),
        Some(ExpertStorageTier::LocalStorage)
    );
    assert_eq!(
        expert_runtime
            .expert_planner
            .location(ExpertId::new(0, deepseek_v4::N_ROUTED_EXPERTS - 1)),
        Some(ExpertStorageTier::LocalStorage)
    );
}

#[test]
fn local_deepseek_v4_tokenizer_loads_from_tokenizer_json_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let tokenizer = TokenizerHandle::load(&model_dir)
        .expect("local DeepSeek V4 tokenizer.json should load without model weights");
    assert_eq!(tokenizer.eos_token_id(), Some(1));
    let ids = tokenizer
        .encode("Ferrule DeepSeek V4 tokenizer smoke")
        .expect("tokenizer should encode a smoke prompt");
    assert!(!ids.is_empty());
    let decoded = tokenizer
        .decode(&ids)
        .expect("tokenizer should decode its own token ids");
    assert!(!decoded.trim().is_empty());
}

fn inventory_tensor<'a>(
    inventory: &'a HfSafetensorsInventory,
    name: &str,
) -> &'a HfSafetensorsTensorInfo {
    inventory
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("missing expected tensor {name}"))
}

fn local_deepseek_v4_dir() -> Option<PathBuf> {
    if let Ok(path) = env::var("FERRULE_DEEPSEEK_V4_DIR") {
        let path = PathBuf::from(path);
        assert!(
            path.join("config.json").exists(),
            "FERRULE_DEEPSEEK_V4_DIR must point at a HF model directory with config.json"
        );
        return Some(path);
    }

    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models")
        .join("DeepSeek-V4-Flash-DSpark");
    default.join("config.json").exists().then_some(default)
}
