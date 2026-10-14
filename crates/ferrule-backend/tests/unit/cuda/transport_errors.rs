use super::{CudaTransferCompletion, CudaTransferDrainError, CudaTransferError};
use ferrule_common::{Error, ErrorChain, IoProtocolError};
use std::error::Error as StdError;
use std::mem::size_of;

#[test]
fn transfer_errors_stay_below_the_large_result_threshold() {
    assert!(size_of::<CudaTransferError>() < 128);
    assert!(size_of::<CudaTransferDrainError>() < 128);
}

#[test]
fn backend_conversion_preserves_the_common_error_source() {
    let error = CudaTransferError::from(Error::from(IoProtocolError::NullSlabAddress));
    assert!(matches!(
        StdError::source(&error).and_then(|source| source.downcast_ref::<Error>()),
        Some(Error::IoProtocol {
            source: IoProtocolError::NullSlabAddress
        })
    ));
    assert_eq!(
        error.to_string(),
        "CUDA async transport: registered slab address must be non-zero"
    );
}

#[test]
fn submission_keeps_primary_and_cleanup_typed_and_separate() {
    for cleanup_failed in [false, true] {
        let error = CudaTransferError::Submission {
            source: Error::from(IoProtocolError::NullSlabAddress),
            cleanup: cleanup_failed
                .then(|| Box::new(Error::from(IoProtocolError::InvalidFenceContract))),
        };
        assert!(matches!(
            StdError::source(&error).and_then(|source| source.downcast_ref::<Error>()),
            Some(Error::IoProtocol {
                source: IoProtocolError::NullSlabAddress
            })
        ));
        let primary = ErrorChain::new(&error)
            .filter_map(|source| source.downcast_ref::<IoProtocolError>())
            .collect::<Vec<_>>();
        assert_eq!(primary, [&IoProtocolError::NullSlabAddress]);
        let CudaTransferError::Submission { cleanup, .. } = &error else {
            unreachable!();
        };
        assert_eq!(
            cleanup
                .as_deref()
                .and_then(|source| { ErrorChain::new(source).downcast_ref::<IoProtocolError>() }),
            cleanup_failed.then_some(&IoProtocolError::InvalidFenceContract)
        );
        let suffix = if cleanup_failed {
            "cleanup fence failed (completion fence identity must be non-zero); custody retained"
        } else {
            "cleanup fence completed"
        };
        assert_eq!(
            error.to_string(),
            format!(
                "CUDA async transport: submission failed (registered slab address must be non-zero); {suffix}"
            )
        );
    }
}

#[test]
fn drain_keeps_completed_identities_and_typed_failure() {
    let id = super::CudaTransferId {
        transport: 1,
        direction: super::CudaTransferDirection::H2D,
        slot: 0,
        generation: 1,
    };
    let error = CudaTransferDrainError {
        completed: vec![CudaTransferCompletion::Canceled { id }],
        source: CudaTransferError::from(Error::from(IoProtocolError::NullSlabAddress)),
    };
    assert_eq!(error.completed, [CudaTransferCompletion::Canceled { id }]);
    assert!(
        StdError::source(&error)
            .and_then(|source| source.downcast_ref::<CudaTransferError>())
            .is_some()
    );
    assert_eq!(
        ErrorChain::new(&error).downcast_ref::<IoProtocolError>(),
        Some(&IoProtocolError::NullSlabAddress)
    );
}
