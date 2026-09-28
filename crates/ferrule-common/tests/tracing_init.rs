//! A separate test executable gives this test exclusive global subscriber ownership.
use ferrule_common::observability::{TracingInitStatus, init_tracing, init_tracing_once, shutdown};

#[test]
fn concurrent_and_repeated_initialization_uses_one_cached_installation() {
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..16).map(|_| scope.spawn(init_tracing_once)).collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), TracingInitStatus::Installed);
        }
    });
    init_tracing();
    shutdown();
    assert_eq!(init_tracing_once(), TracingInitStatus::Installed);
}
