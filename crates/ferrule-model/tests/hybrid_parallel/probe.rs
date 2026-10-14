//! Observations only; never an owner protocol or transaction decision.
use ferrule_model::{
    decoder::*,
    transformer::hybrid_parallel::*,
    transformer::{KvHistory, KvView},
};
use std::collections::BTreeMap;
#[derive(Clone, Debug)]
pub struct Capture {
    pub recurrent: BTreeMap<usize, GatedDeltaNetState>,
    pub kv: BTreeMap<usize, KvHistory>,
}
pub fn capture(
    schema: &HybridStateSchema,
    state: &HybridLayerStates,
    batch: &PackedDecoderBatch,
    view: &CpuKvView,
) -> Capture {
    let end = batch.sequences()[0].sequence_len();
    let mut recurrent = BTreeMap::new();
    let mut kv = BTreeMap::new();
    for entry in schema.entries() {
        match &state.layers()[entry.state_slot] {
            HybridLayerState::GatedDeltaNet(s) => {
                recurrent.insert(entry.global_layer, s.clone());
            }
            HybridLayerState::FullAttention => {
                let HybridLayerSchema::FullAttention {
                    kv_layer,
                    kv_heads,
                    head_dim,
                } = schema.layers()[entry.local_layer]
                else {
                    panic!("schema")
                };
                kv.insert(
                    entry.global_layer,
                    view.history(kv_layer, 0, end - 1, kv_heads, head_dim)
                        .unwrap(),
                );
            }
        }
    }
    Capture { recurrent, kv }
}
pub fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 3e-5, "{a} != {b}");
    }
}
pub fn compare_local(plan: &HybridParallelPlan, lane: usize, local: &Capture, full: &Capture) {
    for layer in plan.layers() {
        let global = layer.global_layer;
        match &layer.attention[lane] {
            HybridLocalAttention::GatedDelta {
                shape,
                qkv_channels,
                value_heads,
                ..
            } => {
                let (a, b) = (&local.recurrent[&global], &full.recurrent[&global]);
                assert_eq!(a.shape(), *shape);
                assert_eq!(a.position(), b.position());
                let conv = qkv_channels
                    .iter()
                    .flat_map(|r| {
                        b.conv_history()[r.start * shape.kernel..r.end * shape.kernel]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<_>>();
                close(a.conv_history(), &conv);
                let width = shape.key_dim * shape.value_dim;
                close(
                    a.recurrent(),
                    &b.recurrent()[value_heads.start * width..value_heads.end * width],
                );
            }
            HybridLocalAttention::Gqa {
                kv_heads, head_dim, ..
            } => {
                let (a, b) = (&local.kv[&global], &full.kv[&global]);
                assert_eq!(a.tokens, b.tokens);
                let global_width = b.key.len() / b.tokens;
                let slice = |values: &[f32]| {
                    values
                        .chunks_exact(global_width)
                        .flat_map(|row| {
                            row[kv_heads.start * head_dim..kv_heads.end * head_dim]
                                .iter()
                                .copied()
                        })
                        .collect::<Vec<_>>()
                };
                close(&a.key, &slice(&b.key));
                close(&a.value, &slice(&b.value));
            }
        }
    }
}
