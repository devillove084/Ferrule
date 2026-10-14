use super::*;
#[test]
fn completion_status_pending_is_not_unknown_but_driver_failure_is() {
    assert!(!completion_is_unknown(CUDA_SUCCESS));
    assert!(!completion_is_unknown(CUDA_ERROR_NOT_READY));
    for code in [1, 3, 700, 719, 999] {
        assert!(completion_is_unknown(code));
    }
}
