The borrowed state is exclusive and cannot escape the sequence's lifetime.

```compile_fail
use ferrule_model::decoder::{GatedDeltaStateRef, StandardSequenceState};
fn escape(state: &mut dyn StandardSequenceState) -> GatedDeltaStateRef<'static> {
    state.linear_state(0).unwrap()
}
```

```compile_fail
use ferrule_model::decoder::{GatedDeltaStateRef, StandardSequenceState};
fn twice(state: &mut dyn StandardSequenceState) -> (GatedDeltaStateRef<'_>, GatedDeltaStateRef<'_>) {
    let first = state.linear_state(0).unwrap();
    let second = state.linear_state(0).unwrap();
    (first, second)
}
```

The contract does not offer cloneable handles or a CPU-only accessor.

```compile_fail
use ferrule_model::decoder::GatedDeltaStateRef;
fn require_clone<T: Clone>() {}
require_clone::<GatedDeltaStateRef<'static>>();
```

```compile_fail
use ferrule_model::decoder::StandardSequenceState;
fn cpu_fallback(state: &mut dyn StandardSequenceState) {
    state.linear_state_mut(0).unwrap();
}
```

CPU extraction is private to the CPU consumer, not a public neutral API.

```compile_fail
use ferrule_model::decoder::GatedDeltaStateRef;
fn cpu(state: &mut GatedDeltaStateRef<'_>) {
    state.cpu().unwrap();
}
```
