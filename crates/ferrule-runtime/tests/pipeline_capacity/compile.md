
The former device boolean is deliberately not a compatibility API:

```compile_fail
use ferrule_runtime::parallel::pipeline::PipelineConfig;
fn old_capacity(config: PipelineConfig) {
    config.physical_pages(true).unwrap();
}
```

CUDA shapes remain ordinary metadata, also in builds without the CUDA feature:

```
use ferrule_model::ModelExecutionBackend;
use ferrule_runtime::parallel::pipeline::{CustodyStrategy, PhysicalKvLayout, PhysicalSlotLimit};
let cuda = PhysicalKvLayout::standard(ModelExecutionBackend::Cuda);
assert_eq!(cuda.custody, CustodyStrategy::Shadowed);
assert_eq!(cuda.slot_limit, PhysicalSlotLimit::I32);
assert_eq!(cuda.physical_pages(8).unwrap(), 16);
```
