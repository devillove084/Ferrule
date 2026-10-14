use super::*;
use crate::args::{Cli, Command};
use clap::Parser;
use ferrule_model::{
    AttentionKind, ModelDescriptor, ModelFamily, MoeSpec, TransformerSemantics, TransformerSpec,
    WeightSource,
};

#[test]
fn parsed_chat_defaults_and_context_reach_requested_effective_report() {
    let config = AutoConfig::from_descriptor(ModelDescriptor {
        path: "metadata-only-no-model".into(),
        spec: TransformerSpec {
            family: ModelFamily::Qwen35,
            architecture: Some("Qwen3_5ForConditionalGeneration".into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(64),
            num_layers: Some(24),
            vocab_size: Some(128),
            num_heads: Some(4),
            num_kv_heads: Some(4),
            head_dim: Some(16),
            attention: AttentionKind::GroupedQuery,
            moe: MoeSpec::none(),
            semantics: TransformerSemantics::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        },
        tensor_classes: Vec::new(),
    });
    for flags in [
        vec!["--backend", "cpu"],
        vec!["--ctx-size", "4096", "--backend", "cpu"],
        vec!["--ctx-size", "8", "--backend", "cpu"],
    ] {
        let mut argv = vec!["ferrule", "chat", "metadata-only-no-model"];
        argv.extend(flags);
        let Command::Chat {
            sampling,
            max_tokens,
            backend,
            chat_template,
            ..
        } = Cli::try_parse_from(argv).unwrap().command
        else {
            panic!("chat parse")
        };
        let generation = sampling.generation_config(max_tokens);
        let options = chat_model_options(&config, &generation, &sampling).unwrap();
        let selection = super::super::backend_policy::select(backend.as_deref()).unwrap();
        let plan = ResidentModelPlanner
            .prepare(&config, selection, chat_template.as_deref(), options)
            .unwrap();
        assert_eq!(plan.requested_backend(), selection);
        assert_eq!(
            plan.requested_options().scheduler_config.prefill_chunk_size,
            4096
        );
        assert_eq!(
            plan.requested_options().driver_config.ctx_size,
            generation.ctx_size
        );
        let effective = plan.effective_options();
        assert_eq!(
            effective.scheduler_config.max_batch_tokens,
            32.min(generation.ctx_size)
        );
        assert_eq!(
            effective.scheduler_config.prefill_chunk_size,
            effective.scheduler_config.max_batch_tokens
        );
        assert!(!effective.driver_config.enable_native_proposals);
        assert!(
            plan.adjustment_report()
                .render()
                .contains("min(32, context)")
        );
        assert!(
            plan.resolution_report()
                .contains("owner live admission required")
        );
        assert_eq!(plan.backend_profile(), "cpu-hybrid-f32-qwen35-0.8b");
    }
}
