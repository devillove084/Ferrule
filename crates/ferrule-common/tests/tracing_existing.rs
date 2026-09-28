//! Isolated from the installation test because subscribers are process-global.
use ferrule_common::observability::{TracingInitStatus, init_tracing, init_tracing_once};

#[test]
fn embedded_subscriber_is_retained_without_panicking() {
    let subscriber = tracing::subscriber::NoSubscriber::default();
    tracing::subscriber::set_global_default(subscriber).unwrap();
    assert_eq!(init_tracing_once(), TracingInitStatus::AlreadyInstalled);
    init_tracing();
    assert_eq!(init_tracing_once(), TracingInitStatus::AlreadyInstalled);
    tracing::dispatcher::get_default(|dispatch| {
        assert!(dispatch.is::<tracing::subscriber::NoSubscriber>());
    });
}
