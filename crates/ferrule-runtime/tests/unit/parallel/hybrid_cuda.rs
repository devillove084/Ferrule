use super::*;
#[test]
fn startup_panic_preserves_typed_unknown_without_a_device() {
    let failure =
        std::panic::catch_unwind(|| cuda_call(|| panic!("fixture lost completion"))).unwrap_err();
    assert_eq!(
        failure.downcast_ref::<PanicQuiescence>(),
        Some(&PanicQuiescence::Unknown)
    );
    assert_eq!(cuda_call(|| 7), 7);
}
