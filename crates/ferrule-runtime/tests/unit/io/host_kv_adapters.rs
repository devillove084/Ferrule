// Included inside the private runtime adapter, not a standalone integration target.
use super::*;

#[test]
fn outcome_classification_is_not_a_physical_certificate() {
    assert_eq!(HostKvSubmission::from(Ok(())), HostKvSubmission::Accepted);
    assert_eq!(
        HostKvSubmission::from(Err(HostKvError::Unknown)),
        HostKvSubmission::Unknown
    );
    for error in [
        HostKvError::ForeignOwner,
        HostKvError::StaleSource,
        HostKvError::InvalidRange,
        HostKvError::InvalidGeometry,
        HostKvError::SharedSource,
        HostKvError::TransactionActive,
        HostKvError::ReaderActive,
        HostKvError::UnknownLease,
        HostKvError::UnknownPlacement,
        HostKvError::InvalidTransition,
        HostKvError::FenceMismatch,
        HostKvError::BudgetExceeded,
        HostKvError::Overflow,
        HostKvError::ProviderUnsupported,
    ] {
        assert_eq!(
            HostKvSubmission::from(Err(error)),
            HostKvSubmission::Rejected(error)
        );
    }
}
