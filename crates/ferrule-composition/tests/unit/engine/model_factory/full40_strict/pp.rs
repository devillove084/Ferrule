//! Private TP1 factory adapter; the numerical/lifecycle harness is shared with TP.
use super::*;
use crate::model_factory::full40_strict::pipeline::{Topology, run};

fn strict(ep: usize) {
    let topology = Topology { pp: 2, tp: 1, ep };
    run(topology, |request, resources| {
        let admission =
            crate::model_factory::qwen35::binding::pipeline_admission(request, &resources).unwrap();
        assert_eq!(
            admission
                .plan
                .segments
                .iter()
                .map(|p| p.layers())
                .collect::<Vec<_>>(),
            [0..20, 20..40]
        );
        assert_eq!(
            admission.plan.segment_layout.as_ref().unwrap().kv_layers(),
            10
        );
        for stage in 0..2 {
            assert_eq!(
                admission
                    .plan
                    .segment_layout
                    .as_ref()
                    .unwrap()
                    .stage_kv_layers(stage),
                5
            );
        }
        assert_eq!(admission.budgets.len(), topology.owners());
        eprintln!("STRICT {topology:?} owner_budgets={:?}", admission.budgets);
        admission.admit_live(&resources).unwrap();
        let bytes = admission.plan.kv.configured_bytes.unwrap();
        let executor = executor(request, resources, admission).unwrap();
        assert_eq!(executor.physical_kv_bytes().unwrap(), bytes);
        executor
    });
}

#[test]
#[ignore = "manual GPU owner only: explicit strict model/reference/devices/report env; full40 F32 all-vocabulary oracle"]
fn pp2_tp1_ep2_full40_f32_strict() {
    strict(2);
}

#[test]
#[ignore = "manual GPU owner only: explicit strict model/reference/devices/report env; full40 F32 all-vocabulary oracle"]
fn pp2_tp1_ep4_full40_f32_strict() {
    strict(4);
}
