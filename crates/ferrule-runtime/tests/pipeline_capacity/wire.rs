use super::*;
use ferrule_model::transformer::LayerSegmentPlan;
use ferrule_runtime::parallel::process::decoder::{
    DECODER_WIRE_VERSION, DecoderBoot, DecoderDevice, DecoderPrecision, DecoderRecipeKind,
    HybridBootFrame, HybridCudaBudgetFrame, HybridLayerFrame, KvConfigFrame, SegmentFrame,
};

fn boot(device: DecoderDevice, hybrid: bool, pages: usize) -> DecoderBoot {
    let c = config(pages);
    DecoderBoot {
        version: DECODER_WIRE_VERSION,
        rank: 0,
        checkpoint: "metadata-only-no-checkpoint-reads".into(),
        recipe: if hybrid {
            DecoderRecipeKind::Qwen35DenseHybrid
        } else {
            DecoderRecipeKind::Synthetic
        },
        segment: SegmentFrame::encode(&LayerSegmentPlan::new(1, 0..1, true, true).unwrap()),
        precision: DecoderPrecision::F32,
        device,
        kv: if hybrid {
            KvConfigFrame::encode_for_hybrid(c, device).unwrap()
        } else {
            KvConfigFrame::encode_for_device(c, device).unwrap()
        },
        experts: None,
        hybrid: hybrid.then(|| HybridBootFrame {
            hidden: 2,
            vocabulary: 4,
            layers: vec![HybridLayerFrame::FullAttention {
                kv_heads: 1,
                head_dim: 2,
            }],
            cuda_budget: match device {
                DecoderDevice::Cpu => None,
                DecoderDevice::Cuda { .. } => Some(HybridCudaBudgetFrame {
                    state_bytes: 1,
                    weight_bytes: 1,
                    workspace_bytes: 1,
                    margin_bytes: 0,
                }),
            },
        }),
    }
}

#[test]
fn standard_hybrid_cpu_cuda_wire_shapes_and_mismatches() {
    for device in [DecoderDevice::Cpu, DecoderDevice::Cuda { ordinal: 7 }] {
        for hybrid in [false, true] {
            let boot = boot(device, hybrid, 7);
            let expected = if hybrid || device != DecoderDevice::Cpu {
                14
            } else {
                7
            };
            let encoded = serde_json::to_value(&boot).unwrap();
            let decoded: DecoderBoot = serde_json::from_value(encoded.clone()).unwrap();
            assert_eq!(decoded, boot);
            assert_eq!(decoded.stage_boot().unwrap().physical_pages, expected);
            assert_eq!(
                decoded.description(2, 4, 1, 2).unwrap().physical_pages,
                expected
            );
            for pages in [0, expected + 1, if expected == 14 { 7 } else { 14 }] {
                let mut wrong = boot.clone();
                wrong.kv.physical_pages = pages;
                assert!(wrong.stage_boot().is_err());
            }
            let mut missing = encoded;
            missing["kv"]
                .as_object_mut()
                .unwrap()
                .remove("physical_pages");
            assert!(serde_json::from_value::<DecoderBoot>(missing).is_err());
        }
    }
}

#[test]
fn cpu_hybrid_above_signed_slot_limit_roundtrips_but_cuda_rejects() {
    let pages = i32::MAX as usize / 2 + 2;
    let cpu = boot(DecoderDevice::Cpu, true, pages);
    assert_eq!(cpu.stage_boot().unwrap().physical_pages, pages * 2);
    assert_eq!(
        cpu.description(2, 4, 1, 2).unwrap().physical_pages,
        pages * 2
    );
    let mut cuda = cpu;
    cuda.device = DecoderDevice::Cuda { ordinal: 0 };
    cuda.hybrid.as_mut().unwrap().cuda_budget = Some(HybridCudaBudgetFrame {
        state_bytes: 1,
        weight_bytes: 1,
        workspace_bytes: 1,
        margin_bytes: 0,
    });
    assert!(cuda.stage_boot().is_err());
    assert!(KvConfigFrame::encode_for_hybrid(config(pages), cuda.device).is_err());
}
