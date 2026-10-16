//! Exercises the real StateDictExpertProvider reservation/materialization bridge.
use super::*;
use crate::transformer::expert_reads::tests::{Fixture, IMAGE, options};
use crate::transformer::{ExpertReadSchedule, StateDictMaterializer};

#[test]
fn default_provider_capability_is_disabled_without_materialization() {
    struct Legacy;
    impl ExpertProvider for Legacy {
        fn expert(&mut self, _: usize, _: usize) -> Result<ExpertAvailability> {
            panic!("must not materialize")
        }
    }
    let f = Fixture::new(1);
    assert!(
        Legacy
            .reserve_expert_upload(&f.metadata[0], 512, 4096)
            .unwrap()
            .is_none()
    );
}

#[test]
fn state_dict_provider_transfers_existing_read_and_aliases_and_preserves_idle_worker() {
    let f = Fixture::new(1);
    let semantics = f
        .metadata
        .iter()
        .map(ExpertSemantics::read_test_fixture)
        .collect::<Vec<_>>();
    let mut materializer = StateDictMaterializer::new(128).unwrap();
    let mut limits = options(ExpertReadSchedule::Lookahead);
    limits.max_host_bytes += 8192;
    materializer.configure_expert_read_pipeline(limits).unwrap();
    let mut provider = StateDictExpertProvider {
        layer: 0,
        bindings: &semantics,
        materializer: &materializer,
        reads: None,
    };
    provider
        .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
        .unwrap();
    let before = materializer.checkpoint_read_counters();
    let mut reservation = provider
        .reserve_expert_upload(&f.metadata[0], 512, 4096)
        .unwrap()
        .unwrap();
    assert_eq!(materializer.checkpoint_read_counters(), before);
    let ExpertAvailability::Ready(expert) = provider.expert(0, 0).unwrap() else {
        panic!()
    };
    provider.transfer_expert_upload(&mut reservation).unwrap();
    assert!(provider.expert(0, 0).is_err());
    drop(expert);
    reservation.prove_completion();
    reservation.release().unwrap();
    drop(provider);
    let observed = materializer.expert_read_pipeline_observation();
    assert_eq!(observed.status, "idle_retained");
    assert_eq!(observed.current_charge.packed_bytes, 0);
    assert_eq!(observed.current_charge.pinned_upload_bytes, 0);
    assert_eq!(observed.stats.workers_created, 1);
}
