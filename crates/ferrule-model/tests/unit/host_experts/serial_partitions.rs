use super::*;

#[test]
fn serial_partition_peak_retains_every_shard_but_reuses_one_temporary() {
    let first = HostCacheMemoryPlan {
        payload_bytes: 100,
        retained_metadata_bytes: 10,
        session_metadata_bytes: 2,
        worker_staging_bytes: 3,
        worker_stack_bytes: 4,
    };
    let second = HostCacheMemoryPlan {
        payload_bytes: 200,
        retained_metadata_bytes: 20,
        session_metadata_bytes: 8,
        worker_staging_bytes: 7,
        worker_stack_bytes: 6,
    };
    let total = HostCacheMemoryPlan::serial_partitions(&[first, second]).unwrap();
    assert_eq!(total.retained_bytes().unwrap(), 330);
    assert_eq!(total.peak_bytes().unwrap(), 351);
    assert_eq!(
        HostCacheMemoryPlan::serial_partitions(&[second, first]).unwrap(),
        total
    );
    assert!(HostCacheMemoryPlan::serial_partitions(&[]).is_err());
    assert!(
        HostCacheMemoryPlan::serial_partitions(&[
            HostCacheMemoryPlan {
                payload_bytes: u64::MAX,
                ..first
            },
            second
        ])
        .is_err()
    );
}
