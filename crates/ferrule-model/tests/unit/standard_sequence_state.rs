//! State-contract tests: no weights, operators or GPU needed for the host cases.
use super::*;
use crate::decoder::{DecoderSequence, GenericDecoderSequenceState};
use crate::execution::{SequenceStateCore, SequenceTopologyId};
use std::{cell::Cell, rc::Rc};

fn schema() -> HybridStateSchema {
    HybridStateSchema {
        layers: vec![
            HybridLayerSchema::GatedDeltaNet(GatedDeltaShape {
                key_heads: 1,
                value_heads: 2,
                key_dim: 3,
                value_dim: 4,
                kernel: 3,
            }),
            HybridLayerSchema::FullAttention {
                kv_layer: 0,
                kv_heads: 1,
                head_dim: 4,
            },
        ],
        entries: vec![
            HybridLayerEntry {
                global_layer: 7,
                local_layer: 0,
                state_slot: 0,
                global_kv_layer: None,
                local_kv_layer: None,
            },
            HybridLayerEntry {
                global_layer: 8,
                local_layer: 1,
                state_slot: 1,
                global_kv_layer: Some(2),
                local_kv_layer: Some(0),
            },
        ],
        range: 7..9,
        total_layers: 9,
    }
}

// Deliberately neither Clone nor Send. Only the typed borrow is implemented;
// there is no CPU-only method obligation or dynamic payload/downcast escape.
struct LocalState {
    state: HybridDecoderSequenceState,
    accesses: Rc<Cell<usize>>,
}
impl DecoderSequence for LocalState {
    fn topology_id(&self) -> SequenceTopologyId {
        self.state.topology_id()
    }
    fn core(&self) -> &SequenceStateCore {
        self.state.core()
    }
    fn core_mut(&mut self) -> &mut SequenceStateCore {
        self.state.core_mut()
    }
}
impl StandardSequenceState for LocalState {
    fn validate_standard_schema(schema: &HybridStateSchema) -> Result<()> {
        HybridDecoderSequenceState::validate_standard_schema(schema)
    }
    fn validate_standard_state_at(
        &self,
        schema: &HybridStateSchema,
        position: usize,
    ) -> Result<()> {
        self.state.validate_standard_state_at(schema, position)
    }
    fn linear_state(&mut self, layer: usize) -> Result<GatedDeltaStateRef<'_>> {
        self.accesses.set(self.accesses.get() + 1);
        self.state.linear_state(layer)
    }
}

#[test]
fn non_clone_local_state_supports_generic_and_object_safe_borrowing() {
    fn validate<S: StandardSequenceState>(state: &S, schema: &HybridStateSchema) {
        S::validate_standard_schema(schema).unwrap();
        state.validate_standard_state(schema).unwrap();
    }
    let schema = schema();
    let accesses = Rc::new(Cell::new(0));
    let mut state = LocalState {
        state: HybridSequenceLifecycle::new(schema.clone())
            .create()
            .unwrap(),
        accesses: Rc::clone(&accesses),
    };
    validate(&state, &schema);
    let original = state.state.kv_state_mut().linear_mut(7).unwrap() as *mut GatedDeltaNetState;
    let neutral: &mut dyn StandardSequenceState = &mut state;
    neutral.validate_standard_state_at(&schema, 0).unwrap();
    {
        let borrowed = neutral.linear_state(7).unwrap();
        #[cfg(not(feature = "cuda"))]
        let GatedDeltaStateRef::Cpu(payload) = borrowed;
        #[cfg(feature = "cuda")]
        let GatedDeltaStateRef::Cpu(payload) = borrowed else {
            panic!("CPU owner must yield CPU storage");
        };
        assert_eq!(payload as *mut GatedDeltaNetState, original);
        payload.position = 1;
    }
    assert!(neutral.validate_standard_state(&schema).is_err());
    neutral.validate_standard_state_at(&schema, 1).unwrap();
    assert_eq!(accesses.get(), 1);
    assert_eq!(
        state.state.kv_state_mut().linear_mut(7).unwrap().position(),
        1
    );
}

#[test]
fn schema_and_global_layer_kind_are_checked_without_a_gpu() {
    let schema = schema();
    let mut lifecycle = HybridSequenceLifecycle::new(schema.clone());
    let mut state = lifecycle.create().unwrap();
    for wrong_layer in [0, 8, 9, usize::MAX] {
        assert!(state.linear_state(wrong_layer).is_err());
    }
    assert!(matches!(
        state.linear_state(7).unwrap(),
        GatedDeltaStateRef::Cpu(_)
    ));
    assert!(state.validate_standard_state_at(&schema, 1).is_err());
    let mut wrong_schema = schema.clone();
    wrong_schema.total_layers += 1;
    assert!(state.validate_standard_state(&wrong_schema).is_err());
    state.kv_state_mut().layers.swap(0, 1);
    assert!(state.validate_standard_state(&schema).is_err());
    assert!(state.linear_state(7).is_err());
    // Release must still accept dirty/partially executed working state.
    lifecycle.try_release(state).unwrap();
}

#[test]
fn stateless_state_rejects_recurrence_but_accepts_attention_schema() {
    let mut schema = schema();
    let mut state = GenericDecoderSequenceState::new((), ());
    assert!(GenericDecoderSequenceState::validate_standard_schema(&schema).is_err());
    assert!(state.validate_standard_state(&schema).is_err());
    assert!(state.linear_state(7).is_err());
    schema.layers[0] = schema.layers[1].clone();
    GenericDecoderSequenceState::validate_standard_schema(&schema).unwrap();
    state.validate_standard_state(&schema).unwrap();
    assert!(state.linear_state(8).is_err());
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::decoder::{
        CudaHybridSequenceLifecycle, CudaHybridSequenceState, HybridCudaDevice,
        HybridCudaMemoryBudget,
    };
    use crate::execution::ExecutionPrecisionPolicy;
    use ferrule_common::execution::ExecutionTransactionId;

    fn device() -> HybridCudaDevice {
        HybridCudaDevice::new_on_device(
            0,
            HybridCudaMemoryBudget {
                kv_pages: 1,
                state_bytes: 1 << 20,
                weight_bytes: 1 << 20,
                workspace_bytes: 1 << 20,
            },
        )
        .unwrap()
    }

    #[test]
    fn cuda_schema_validation_does_not_construct_an_owner() {
        CudaHybridSequenceState::validate_standard_schema(&schema()).unwrap();
    }

    #[test]
    #[ignore = "requires CUDA GPU; F32/BF16 typed state identity and wrong owner/kind"]
    fn physical_borrow_preserves_kind_owner_and_release_custody() {
        let owner = device();
        let foreign = device();
        for precision in [
            ExecutionPrecisionPolicy::f32(),
            ExecutionPrecisionPolicy::bf16_compatibility(),
        ] {
            let schema = schema();
            let mut lifecycle = CudaHybridSequenceLifecycle::with_precision(
                schema.clone(),
                owner.clone(),
                precision,
            )
            .unwrap();
            let mut wrong_owner = CudaHybridSequenceLifecycle::with_precision(
                schema.clone(),
                foreign.clone(),
                precision,
            )
            .unwrap();
            let mut state = lifecycle.create().unwrap();
            state.validate_standard_state(&schema).unwrap();
            assert!(state.linear_state(8).is_err());
            assert!(state.linear_state(0).is_err());
            let bytes = owner.live_state_bytes();
            let expected = if precision == ExecutionPrecisionPolicy::f32() {
                state.kv_state().layers()[0].as_ref().unwrap() as *const _ as usize
            } else {
                state.kv_state().bf16_layers().unwrap()[0].as_ref().unwrap() as *const _ as usize
            };
            match state.linear_state(7).unwrap() {
                GatedDeltaStateRef::Cuda(payload) => {
                    assert_eq!(precision, ExecutionPrecisionPolicy::f32());
                    assert_eq!(payload as *const _ as usize, expected);
                    payload.validate_owner(owner.operators()).unwrap();
                    assert!(payload.validate_owner(foreign.operators()).is_err());
                }
                GatedDeltaStateRef::CudaBf16(payload) => {
                    assert_eq!(precision, ExecutionPrecisionPolicy::bf16_compatibility());
                    assert_eq!(payload as *const _ as usize, expected);
                    payload.validate_owner(owner.operators()).unwrap();
                    assert!(payload.validate_owner(foreign.operators()).is_err());
                }
                GatedDeltaStateRef::Cpu(_) => panic!("no CPU fallback"),
            }
            assert_eq!(owner.live_state_bytes(), bytes);
            let request =
                DecoderSequenceCheckout::new(ExecutionTransactionId::new(1).unwrap(), 0, 0);
            assert!(wrong_owner.checkout(request, &state).is_err());
            let topology = state.topology_id();
            let (_, state) = wrong_owner.try_release(state).unwrap_err().into_parts();
            assert_eq!(state.topology_id(), topology);
            let wrong_precision = if precision == ExecutionPrecisionPolicy::f32() {
                ExecutionPrecisionPolicy::bf16_compatibility()
            } else {
                ExecutionPrecisionPolicy::f32()
            };
            let mut wrong_kind = CudaHybridSequenceLifecycle::with_precision(
                schema.clone(),
                owner.clone(),
                wrong_precision,
            )
            .unwrap();
            assert!(wrong_kind.checkout(request, &state).is_err());
            let (_, state) = wrong_kind.try_release(state).unwrap_err().into_parts();
            assert_eq!(state.topology_id(), topology);
            assert_eq!(owner.live_state_bytes(), bytes);
            lifecycle.try_release(state).unwrap();
            assert_eq!(owner.live_state_bytes(), 0);
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU; Unknown completion cannot reclaim F32/BF16 credits"]
    fn quarantined_owner_rejects_borrows_checkout_and_release_without_reclaiming_credits() {
        for precision in [
            ExecutionPrecisionPolicy::f32(),
            ExecutionPrecisionPolicy::bf16_compatibility(),
        ] {
            let owner = device();
            let mut lifecycle =
                CudaHybridSequenceLifecycle::with_precision(schema(), owner.clone(), precision)
                    .unwrap();
            let mut state = lifecycle.create().unwrap();
            let bytes = owner.live_state_bytes();
            assert!(bytes > 0);
            owner.quarantine();
            assert!(state.linear_state(7).is_err());
            let request =
                DecoderSequenceCheckout::new(ExecutionTransactionId::new(1).unwrap(), 0, 0);
            assert!(lifecycle.checkout(request, &state).is_err());
            let (_, state) = lifecycle.try_release(state).unwrap_err().into_parts();
            assert_eq!(owner.live_state_bytes(), bytes);
            drop(state);
            assert_eq!(owner.live_state_bytes(), bytes);
        }
    }
}
