#![cfg(all(unix, not(feature = "cpu-reference")))]

use ferrule_composition::BUILTIN_PROCESS_BINDINGS;
use ferrule_runtime::parallel::process::decoder::{
    DECODER_WIRE_VERSION, DecoderBoot, DecoderDevice, DecoderPrecision, DecoderRecipeKind,
    KvConfigFrame, SegmentFrame,
};

fn boot() -> DecoderBoot {
    DecoderBoot {
        version: DECODER_WIRE_VERSION,
        rank: 7,
        checkpoint: "not-opened".into(),
        recipe: DecoderRecipeKind::Synthetic,
        segment: SegmentFrame {
            total_layers: 2,
            layers: 0..2,
            embedding: true,
            output: true,
        },
        precision: DecoderPrecision::F32,
        device: DecoderDevice::Cpu,
        kv: KvConfigFrame {
            page_size: 2,
            max_pages: 4,
            physical_pages: 4,
            max_positions: 16,
            max_batch_tokens: 4,
            session_capacity: 1,
            max_parameter_bytes: 4096,
            max_ack_polls: 8,
        },
        experts: None,
        hybrid: None,
    }
}

#[test]
fn production_binding_does_not_enable_cpu_or_synthetic_execution_in_default_builds() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    let mut boot = boot();
    boot.checkpoint = dir.path().into();
    for recipe in [
        DecoderRecipeKind::Synthetic,
        DecoderRecipeKind::Qwen3Dense,
        DecoderRecipeKind::Qwen3Moe,
    ] {
        boot.recipe = recipe;
        let error = (BUILTIN_PROCESS_BINDINGS.decoder)(&boot).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("CPU process execution requires cpu-reference"),
            "{error}"
        );
    }
}
