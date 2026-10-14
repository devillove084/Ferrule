The committed adapter owns exclusive manager/backend/sequence borrows until it
is finished and dropped. It is not an entered append transaction view.

This signature check only compiles; it creates no CUDA context or work:

```no_run
use ferrule_runtime::{KvPageManager, io::host_kv_adapters::LiveCommittedGqa};
use ferrule_model::decoder::{DecoderSequence, DecoderKvSequenceCustody, PagedKvBackend, TypedCudaPagedKvPool};
use ferrule_model::transformer::{CudaStandardDecoderOperators, CommittedGqaRequest};
use ferrule_common::host_kv::{CommittedKvSource, placement::PlacementBudget};
fn begin<'a, S: DecoderSequence>(manager: &'a mut KvPageManager,
    operators: &mut CudaStandardDecoderOperators,
    backend: &'a mut PagedKvBackend<TypedCudaPagedKvPool<S>>, sequence: &'a mut S,
    custody: &DecoderKvSequenceCustody, source: CommittedKvSource,
    request: CommittedGqaRequest, budget: &PlacementBudget,
) -> ferrule_common::Result<LiveCommittedGqa<'a, S>> {
    LiveCommittedGqa::begin(manager, operators, backend, sequence, custody, source, request, budget)
}
```

The manager cannot admit append while the live operation still borrows it:

```compile_fail,E0499
use ferrule_runtime::{KvPageManager, io::host_kv_adapters::LiveCommittedGqa};
use ferrule_model::decoder::{DecoderSequence, DecoderKvSequenceCustody, PagedKvBackend, TypedCudaPagedKvPool};
use ferrule_model::transformer::{CudaStandardDecoderOperators, CommittedGqaRequest};
use ferrule_common::host_kv::{CommittedKvSource, placement::PlacementBudget};
fn overlap<S: DecoderSequence>(manager: &mut KvPageManager,
    operators: &mut CudaStandardDecoderOperators,
    backend: &mut PagedKvBackend<TypedCudaPagedKvPool<S>>, sequence: &mut S,
    custody: &DecoderKvSequenceCustody, source: CommittedKvSource,
    request: CommittedGqaRequest, budget: &PlacementBudget,
) -> ferrule_common::Result<()> {
    let mut live = LiveCommittedGqa::begin(manager, operators, backend, sequence, custody, source, request, budget)?;
    let _ = manager.reserve(custody.page_state_slot, custody.page_generation, 1);
    live.poll()?;
    Ok(())
}
```

A physical pool mutation also cannot overlap that borrow:

```compile_fail,E0499
use ferrule_runtime::{KvPageManager, io::host_kv_adapters::LiveCommittedGqa};
use ferrule_model::decoder::{DecoderSequence, DecoderKvSequenceCustody, DecoderKvBackend, PagedKvBackend, TypedCudaPagedKvPool};
use ferrule_model::transformer::{CudaStandardDecoderOperators, CommittedGqaRequest};
use ferrule_common::host_kv::{CommittedKvSource, placement::PlacementBudget};
fn overlap<S: DecoderSequence>(manager: &mut KvPageManager,
    operators: &mut CudaStandardDecoderOperators,
    backend: &mut PagedKvBackend<TypedCudaPagedKvPool<S>>, sequence: &mut S,
    custody: &DecoderKvSequenceCustody, source: CommittedKvSource,
    request: CommittedGqaRequest, budget: &PlacementBudget,
) -> ferrule_common::Result<()> {
    let pages = source.pages.clone();
    let mut live = LiveCommittedGqa::begin(manager, operators, backend, sequence, custody, source, request, budget)?;
    backend.release(&pages)?;
    live.poll()?;
    Ok(())
}
```
