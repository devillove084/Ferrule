use ferrule_runtime::{HostKvDecodeConfig, KvExecutionPolicy};
#[test]
fn resident_is_default_and_host_kv_has_explicit_bounded_scratch() {
    assert_eq!(KvExecutionPolicy::default(), KvExecutionPolicy::Resident);
    let config = HostKvDecodeConfig::default();
    assert!(config.block_tokens > 0);
    assert!(config.capacity.pageable_host_bytes > 0);
    assert!(config.capacity.pinned_host_bytes > 0);
    assert!(config.capacity.staging_bytes > 0);
    assert_ne!(
        KvExecutionPolicy::HostKvDecode(config),
        KvExecutionPolicy::Resident
    );
}

#[test]
fn idle_reclaim_is_explicit_and_budget_domains_are_independent() {
    let config = ferrule_runtime::IdleKvReclaimConfig {
        staging_page_limit: 2,
        device_bytes: 4096,
        host_bytes: 2048,
        pinned_bytes: 2048,
    };
    let policy = KvExecutionPolicy::IdleReclaim(config);
    assert_ne!(policy, KvExecutionPolicy::Resident);
    assert_ne!(
        policy,
        KvExecutionPolicy::HostKvDecode(HostKvDecodeConfig::default())
    );
}
