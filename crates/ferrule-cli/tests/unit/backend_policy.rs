use super::*;

#[test]
fn default_and_auto_are_cuda_even_in_reference_builds() {
    for value in [None, Some("auto"), Some("AUTO"), Some("cuda")] {
        assert_eq!(select(value).unwrap(), BackendSelection::Cuda);
    }
    // Library policy is deliberately unchanged.
    assert_eq!(BackendSelection::default(), BackendSelection::Auto);
}

#[test]
fn cpu_aliases_require_the_explicit_reference_feature() {
    for value in ["cpu", "cpu-reference", "reference"] {
        let result = select(Some(value));
        if cfg!(feature = "cpu-reference") {
            assert_eq!(result.unwrap(), BackendSelection::Cpu);
        } else {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("test-only"));
            assert!(error.contains("--features cpu-reference"));
        }
    }
    assert!(select(Some("unsupported")).is_err());
}
