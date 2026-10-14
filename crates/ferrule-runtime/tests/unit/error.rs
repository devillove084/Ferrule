use super::Error;

#[test]
fn primary_and_cleanup_api_keep_cleanup_out_of_primary() {
    let primary = Error::Invariant {
        message: "driver execution failed".into(),
    };
    let cleanup = Error::Invariant {
        message: "cannot release an unknown paged KV page".into(),
    };
    let error = Error::Cleanup {
        operation: "resident transaction abort",
        source: Box::new(primary),
        cleanup: Box::new(cleanup),
    };

    assert!(
        matches!(error.primary(), Error::Invariant { message } if message == "driver execution failed")
    );
    assert_eq!(error.cleanup_errors().len(), 1);
    assert!(
        error.cleanup_errors()[0]
            .to_string()
            .contains("unknown paged KV page")
    );
}
