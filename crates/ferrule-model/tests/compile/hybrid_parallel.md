## Production/reference boundary

The geometry and owner-local physical layout compile without CPU execution:

```no_run
use ferrule_model::transformer::hybrid_parallel::HybridParallelPlan;
fn metadata(plan: &HybridParallelPlan, segment: &ferrule_model::transformer::LayerSegmentPlan) {
    let layout = plan.rank_layout(0).unwrap();
    let _ = layout.state_bytes().unwrap();
    let _ = plan.expert_attachment();
    let schedule = plan.collective_schedule(segment).unwrap();
    for site in schedule.wire_sites() {
        let _ = site.shape.elements(2, 5).unwrap();
    }
}
```

CPU scoped bindings and prepared execution remain explicitly opt-in:

```compile_fail
use ferrule_model::transformer::hybrid_parallel::HybridExpertCallBinding;
```

```compile_fail
use ferrule_model::transformer::HybridDecoderSegment;
let _ = HybridDecoderSegment::execute_with_experts;
```

```compile_fail
use ferrule_model::transformer::hybrid_parallel::HybridTensorCpuSegment;
let _ = HybridTensorCpuSegment::prepare;
```
