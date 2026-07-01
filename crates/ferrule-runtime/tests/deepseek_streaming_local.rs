use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};

use ferrule_model::{
    families::{AttentionTensorKind, HyperConnectionStage, RouterTensorKind},
    HfSafetensorsInventory, HfSafetensorsTensorInfo, ModelFamily, TensorRole,
};
use ferrule_runtime::{
    bind_attention_from_hf, bind_hyper_connection_from_hf, bind_hyper_connection_head_from_hf,
    bind_router_from_hf, bind_shared_swiglu_ffn_from_hf, ExpertComputeBundle, ExpertId,
    ExpertLinearFormat, ExpertLoadReason, ExpertSource, ExpertStreamingPlanner,
    ExpertStreamingPolicy, ExpertStreamingReader, HyperConnectionConfig, SourceLinearFormat,
    SourceLinearPayload, SourceTensorReader, SourceTensorSlice, TokenizerHandle,
};

#[test]
fn local_deepseek_v4_expert_streaming_reads_one_selected_expert_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let routed = inventory.routed_expert_tensors(&ModelFamily::DeepSeekV4);
    assert_eq!(routed.len(), 66_048);

    let unique_experts = routed
        .iter()
        .map(|tensor| (tensor.descriptor.layer, tensor.descriptor.expert))
        .collect::<BTreeSet<_>>();
    assert_eq!(unique_experts.len(), 43 * 256);

    let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(6));
    assert_eq!(
        planner
            .register_hf_routed_expert_tensor_sets(&model_dir, routed)
            .expect("expert tensor sets should register"),
        43 * 256
    );

    let step = planner
        .plan_layer_step(0, &[0, 1, 2, 3, 4, 5], &[])
        .expect("selected experts should be plannable");
    assert_eq!(step.loads.len(), 6);
    assert!(step
        .loads
        .iter()
        .all(|load| load.reason == ExpertLoadReason::Selected));

    let load = step
        .loads
        .iter()
        .find(|load| load.expert == ExpertId::new(0, 0))
        .expect("layer 0 expert 0 should be selected");
    let ExpertSource::LocalTensorSet { tensors } = &load.source else {
        panic!("expected LocalTensorSet for HF expert source");
    };
    assert_eq!(tensors.len(), 6);
    assert!(tensors.iter().all(|tensor| tensor.path.exists()));
    assert!(tensors.iter().all(|tensor| tensor.bytes > 0));

    let reader = ExpertStreamingReader::new(64 * 1024 * 1024);
    let payload = reader
        .read_source(load.expert, &load.source)
        .expect("bounded reader should read one selected expert from local shards");
    assert_eq!(payload.expert, ExpertId::new(0, 0));
    assert_eq!(payload.tensors.len(), 6);
    assert_eq!(
        payload
            .tensors
            .iter()
            .map(|tensor| tensor.bytes.len() as u64)
            .sum::<u64>(),
        load.source.bytes()
    );
    assert!(payload
        .tensors
        .iter()
        .any(|tensor| tensor.slice.dtype == "I8"));
    assert!(payload
        .tensors
        .iter()
        .any(|tensor| tensor.slice.dtype == "F8_E8M0"));
    let bundle = ExpertComputeBundle::from_source_payload(payload)
        .expect("source slices should form a FP4 expert compute bundle");
    assert_eq!(
        bundle.gate.format,
        ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features: 2048,
            in_features: 4096,
            block_size: 32,
        }
    );
    assert_eq!(
        bundle.down.format,
        ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features: 4096,
            in_features: 2048,
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
    let shared = inventory.shared_expert_tensors(&ModelFamily::DeepSeekV4);
    assert_eq!(shared.len(), 43 * 3 * 2);

    let ffn = bind_shared_swiglu_ffn_from_hf(
        &model_dir,
        0,
        &shared,
        &SourceTensorReader::new(64 * 1024 * 1024),
        10.0,
    )
    .expect("layer 0 shared expert should bind into source linears");
    assert_eq!(ffn.gate.format.in_features(), 4096);
    assert_eq!(ffn.gate.format.out_features(), 2048);
    assert_eq!(ffn.up.format.in_features(), 4096);
    assert_eq!(ffn.up.format.out_features(), 2048);
    assert_eq!(ffn.down.format.in_features(), 2048);
    assert_eq!(ffn.down.format.out_features(), 4096);
}

#[test]
fn local_deepseek_v4_router_binds_hash_and_score_layers_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let routers = inventory.router_tensors(&ModelFamily::DeepSeekV4);
    assert_eq!(
        routers
            .iter()
            .filter(|tensor| tensor.descriptor.kind == RouterTensorKind::Weight)
            .count(),
        43
    );
    assert_eq!(
        routers
            .iter()
            .filter(|tensor| tensor.descriptor.kind == RouterTensorKind::HashTable)
            .count(),
        3
    );
    assert_eq!(
        routers
            .iter()
            .filter(|tensor| tensor.descriptor.kind == RouterTensorKind::Bias)
            .count(),
        40
    );

    let reader = SourceTensorReader::new(64 * 1024 * 1024);
    let hash_router = bind_router_from_hf(&model_dir, 0, &routers, &reader)
        .expect("layer 0 hash router should bind");
    assert_eq!(hash_router.weight.format.out_features(), 256);
    assert_eq!(hash_router.weight.format.in_features(), 4096);
    assert!(hash_router.bias.is_none());
    let hash = hash_router
        .hash_experts_for_token(0)
        .expect("token 0 should be in hash table")
        .expect("layer 0 should have a hash table");
    assert_eq!(hash.len(), 6);
    assert!(hash.iter().all(|expert| *expert < 256));

    let score_router = bind_router_from_hf(&model_dir, 3, &routers, &reader)
        .expect("layer 3 score router should bind");
    assert_eq!(score_router.weight.format.out_features(), 256);
    assert_eq!(score_router.weight.format.in_features(), 4096);
    assert_eq!(score_router.bias.as_ref().map(Vec::len), Some(256));
    assert!(score_router.hash_table.is_none());
}

#[test]
fn local_deepseek_v4_source_linear_reads_attention_fp8_pair_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let weight = inventory_tensor(&inventory, "layers.0.attn.wq_a.weight");
    let scale = inventory_tensor(&inventory, "layers.0.attn.wq_a.scale");
    assert_eq!(weight.dtype, "F8_E4M3");
    assert_eq!(scale.dtype, "F8_E8M0");

    let reader = SourceTensorReader::new(64 * 1024 * 1024);
    let weight = reader
        .read_slice(&SourceTensorSlice::from_hf_inventory(&model_dir, weight))
        .expect("wq_a weight should be readable as a bounded source tensor");
    let scale = reader
        .read_slice(&SourceTensorSlice::from_hf_inventory(&model_dir, scale))
        .expect("wq_a scale should be readable as a bounded source tensor");
    let linear = SourceLinearPayload::from_weight_and_scale(
        TensorRole::AttentionLatentQueryA,
        weight,
        Some(scale),
    )
    .expect("wq_a weight/scale should infer an FP8 source linear format");
    assert_eq!(
        linear.format,
        SourceLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: 1024,
            in_features: 4096,
            block_m: 128,
            block_k: 128,
        }
    );
}

#[test]
fn local_deepseek_v4_attention_and_hc_bind_real_sources_if_present() {
    let Some(model_dir) = local_deepseek_v4_dir() else {
        return;
    };

    let inventory = HfSafetensorsInventory::open(&model_dir, ModelFamily::DeepSeekV4)
        .expect("local DeepSeek V4 inventory should parse headers only");
    let attention = inventory.attention_tensors(&ModelFamily::DeepSeekV4);
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
    assert_eq!(core_attention, 43 * 13);
    assert_eq!(compressor, 164);
    assert_eq!(indexer, 147);

    let reader = SourceTensorReader::new(64 * 1024 * 1024);
    let layer0 = bind_attention_from_hf(&model_dir, 0, &attention, &reader)
        .expect("layer 0 attention source payload should bind");
    assert_eq!(
        layer0.wq_a.format,
        SourceLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: 1024,
            in_features: 4096,
            block_m: 128,
            block_k: 128,
        }
    );
    assert_eq!(
        layer0.wq_b.format,
        SourceLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: 32768,
            in_features: 1024,
            block_m: 128,
            block_k: 128,
        }
    );
    assert_eq!(
        layer0.wkv.format,
        SourceLinearFormat::Fp8E4M3WithE8M0Scale {
            out_features: 512,
            in_features: 4096,
            block_m: 128,
            block_k: 128,
        }
    );
    assert_eq!(layer0.q_norm.len(), 1024);
    assert_eq!(layer0.kv_norm.len(), 512);
    assert_eq!(layer0.attention_sink.len(), 64);
    assert!(layer0.auxiliary.is_empty());

    let layer2 = bind_attention_from_hf(&model_dir, 2, &attention, &reader)
        .expect("layer 2 attention source payload should bind with compressor/indexer auxiliary");
    assert!(!layer2.auxiliary.is_empty());

    let hc = inventory.hyper_connection_tensors(&ModelFamily::DeepSeekV4);
    assert_eq!(hc.len(), 43 * 6 + 3);
    let config = HyperConnectionConfig {
        hc_mult: 4,
        hidden_size: 4096,
        sinkhorn_iters: 4,
        eps: 1e-6,
        norm_eps: 1e-6,
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
    assert_eq!(hc_attn.function.len(), 24 * 16384);
    assert_eq!(hc_attn.scale.len(), 3);
    assert_eq!(hc_attn.base.len(), 24);

    let hc_ffn = bind_hyper_connection_from_hf(
        &model_dir,
        0,
        HyperConnectionStage::FeedForward,
        &hc,
        &reader,
        config,
    )
    .expect("layer 0 FFN HC should bind");
    assert_eq!(hc_ffn.function.len(), 24 * 16384);
    assert_eq!(hc_ffn.scale.len(), 3);
    assert_eq!(hc_ffn.base.len(), 24);

    let hc_head = bind_hyper_connection_head_from_hf(&model_dir, &hc, &reader, config)
        .expect("global HC head should bind");
    assert_eq!(hc_head.function.len(), 4 * 16384);
    assert_eq!(hc_head.scale.len(), 1);
    assert_eq!(hc_head.base.len(), 4);
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
