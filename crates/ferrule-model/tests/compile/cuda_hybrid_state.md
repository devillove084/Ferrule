```compile_fail
use ferrule_model::decoder::CudaHybridSequenceState;
fn require_clone<T: Clone>() {}
require_clone::<CudaHybridSequenceState>();
```

CUDA physical state and borrowed handles stay on their owner thread.

```compile_fail
use ferrule_model::decoder::CudaHybridSequenceState;
fn require_send<T: Send>() {}
require_send::<CudaHybridSequenceState>();
```

```compile_fail
use ferrule_model::decoder::GatedDeltaStateRef;
fn require_send<T: Send>() {}
require_send::<GatedDeltaStateRef<'static>>();
```
