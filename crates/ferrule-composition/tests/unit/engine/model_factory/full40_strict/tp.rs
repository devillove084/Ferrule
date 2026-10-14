//! Private TP2 factory adapter; no alternate forward or public test-only API.
use super::*;
use crate::model_factory::full40_strict::pipeline::{Topology, run};

fn strict(pp: usize) {
    let topology = Topology { pp, tp: 2, ep: 2 };
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
            (0..pp)
                .map(|s| s * (40 / pp)..(s + 1) * (40 / pp))
                .collect::<Vec<_>>()
        );
        for stage in admission.plan.layout.stages() {
            assert_eq!(
                stage.schema().kv_planes(16, 32).unwrap().planes()[0].layer_count,
                10 / pp
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
fn pp1_tp2_ep2_full40_f32_strict() {
    strict(1);
}

#[test]
#[ignore = "manual GPU owner only: explicit strict model/reference/devices/report env; full40 F32 all-vocabulary oracle"]
fn pp2_tp2_ep2_full40_f32_strict() {
    strict(2);
}
