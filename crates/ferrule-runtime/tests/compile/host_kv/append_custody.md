A borrowed prefix is not detached metadata: logical mutation cannot occur while
that capability will be used again (including by the CUDA binding/output guard).

```compile_fail,E0502,E0505
use ferrule_runtime::{KvPageManager, KvReservation};
use ferrule_common::execution::ExecutionTransactionId;
fn abort_while_borrowed(manager: &mut KvPageManager, reservation: KvReservation) {
    let prefix = manager.derive_append_prefix(
        &reservation, ExecutionTransactionId::new(1).unwrap(), 0,
    ).unwrap();
    manager.abort_reservations(vec![reservation]).unwrap();
    prefix.validate().unwrap();
}
```
