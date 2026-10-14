//! Private numeric-domain operations used only by the ignored CPU benchmark.
use super::*;
use std::hint::black_box;
use std::time::Instant;

fn measure<T>(mut f: impl FnMut() -> T) -> u128 {
    for _ in 0..4 {
        black_box(f());
    }
    let mut samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..8 {
            black_box(f());
        }
        samples.push(start.elapsed().as_nanos() / 8);
    }
    samples.sort_unstable();
    samples[2]
}

pub(crate) fn costs(owner: usize, frames: &[NumericToken]) -> serde_json::Value {
    let owner = owner as u32 + 1;
    let entries = (0..40)
        .flat_map(|l| (0..256).map(move |e| (l, e, e as u32 % 8 + 1)))
        .collect::<Vec<_>>();
    let expected = entries
        .iter()
        .filter(|(_, _, o)| *o == owner)
        .map(|&(layer, expert, _)| NumericSource {
            layer,
            expert,
            identity: [1; 32],
            content_hash: [2; 32],
            encoding: 1,
            generation: 1,
            bytes: 128,
        })
        .collect::<Vec<_>>();
    let boot = NumericBoot {
        numeric_expert_boot: NUMERIC_EXPERT_WIRE_VERSION,
        numeric_profile: ferrule_model::transformer::NumericFp8Precision::expert_wire_default(),
        checkpoint: "unused-codec-metadata".into(),
        total_layers: 40,
        layers: 0..40,
        placement: ExpertPlacementFrame {
            source_scope: ExpertSourceScope::ExternalStage,
            source: 0,
            members: (1..=8).collect(),
            entries,
            max_tokens: 256,
            max_bytes: 256 * 2048 * 4,
            devices: vec![DecoderDevice::Cuda { ordinal: 0 }; 8],
            timeout_ms: 300000,
        },
        owner,
        hidden: 2048,
        max_rows: 32,
        max_parameter_bytes: 4096,
        host_max_bytes: 1 << 20,
        host_max_experts: 1280,
        host_workers: 1,
        host_headroom_bytes: 0,
        device_bytes: 1 << 20,
        workspace_bytes: 4096,
        expected,
    };
    let context = ExpertDispatchContext {
        transaction: ExecutionTransactionId::new(1).unwrap(),
        source_rank: ParallelRankId::new(0),
        layer: frames[0].layer,
    };
    let tokens = frames
        .iter()
        .cloned()
        .map(NumericToken::decode)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let bucket = ExpertTokenBucket {
        owner_rank: ParallelRankId::new(owner),
        tokens,
    };
    validate_token_frames(&boot, context, frames).unwrap();
    let encode = measure(|| {
        bucket
            .tokens
            .iter()
            .map(NumericToken::encode)
            .collect::<Vec<_>>()
    });
    let validate =
        measure(|| validate_token_frames(black_box(&boot), context, black_box(frames)).unwrap());
    let reply_encode = measure(|| {
        bucket
            .tokens
            .iter()
            .map(|t| {
                NumericToken::encode_values(t, t.payload.iter().map(|v| v.to_bits()).collect())
            })
            .collect::<Vec<_>>()
    });
    let mut samples = Vec::new();
    for _ in 0..5 {
        let inputs = (0..8).map(|_| frames.to_vec()).collect::<Vec<_>>();
        let start = Instant::now();
        for input in inputs {
            black_box(validate_tokens(&boot, context, input).unwrap());
        }
        samples.push(start.elapsed().as_nanos() / 8);
    }
    samples.sort_unstable();
    serde_json::json!({"numeric_encode":encode,"numeric_validate_proxy":validate,"numeric_validate_child":samples[2],"numeric_reply_encode":reply_encode})
}
