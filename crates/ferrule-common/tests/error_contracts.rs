use std::any::TypeId;
use std::error::Error as StdError;
use std::fmt;
use std::io;
use std::sync::Arc;

use ferrule_common::{
    ClassifiedError, Error, ErrorChain, ErrorClass, WorkerExecutionError, WorkerRequestError,
};

#[derive(Debug)]
struct Leaf(&'static str);

impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl StdError for Leaf {}

fn leaf(message: &'static str) -> Error {
    Error::Backend {
        source: Box::new(Leaf(message)),
    }
}

fn labels(error: &(dyn StdError + 'static)) -> Vec<&'static str> {
    ErrorChain::new(error)
        .filter_map(|cause| cause.downcast_ref::<Leaf>().map(|leaf| leaf.0))
        .collect()
}

#[test]
fn primary_cleanup_and_nested_batches_keep_all_sources_in_order() {
    let error = Error::context(
        "outer",
        Error::with_cleanup(
            "primary operation",
            Error::context("primary context", leaf("primary")),
            Error::failures(
                "cleanup steps",
                vec![
                    leaf("cleanup first"),
                    Error::with_cleanup(
                        "nested cleanup",
                        leaf("cleanup second"),
                        Err(leaf("cleanup third")),
                    ),
                    Error::failures("nested batch", vec![leaf("cleanup fourth")]).unwrap_err(),
                ],
            ),
        ),
    );
    assert_eq!(
        labels(&error),
        [
            "primary",
            "cleanup first",
            "cleanup second",
            "cleanup third",
            "cleanup fourth"
        ]
    );

    // The standard source chain still follows ONLY the primary cause. Secondary
    // branches are available explicitly, not substituted as primary policy.
    let mut cause: &(dyn StdError + 'static) = &error;
    let mut primary = Vec::new();
    while let Some(source) = cause.source() {
        if let Some(leaf) = source.downcast_ref::<Leaf>() {
            primary.push(leaf.0);
        }
        cause = source;
    }
    assert_eq!(primary, ["primary"]);
    let Error::Context { source, .. } = &error else {
        panic!("context lost")
    };
    let Error::Cleanup { cleanup, .. } = source.as_ref() else {
        panic!("cleanup lost")
    };
    let Error::FailureBatch { failures, .. } = cleanup.as_ref() else {
        panic!("batch lost")
    };
    assert_eq!(failures.len(), 3);
    assert_eq!(labels(&failures[1]), ["cleanup second", "cleanup third"]);
}

#[test]
fn success_cleanup_and_empty_batches_do_not_change_primary() {
    let error = Error::with_cleanup("not emitted", leaf("primary"), Ok(()));
    assert!(matches!(error, Error::Backend { .. }));
    assert_eq!(labels(&error), ["primary"]);
    assert!(Error::failures("empty", vec![]).is_ok());
    let single = Error::failures("single", vec![leaf("one")]).unwrap_err();
    assert!(matches!(single, Error::FailureBatch { .. }));
    assert_eq!(
        single.to_string(),
        "single encountered 1 independent failures"
    );
}

#[test]
fn existing_internal_display_contracts_remain_stable() {
    let context = Error::context("load", leaf("primary"));
    assert_eq!(context.to_string(), "load: backend: primary");
    let cleanup = Error::with_cleanup("load", context, Err(leaf("cleanup")));
    assert_eq!(
        cleanup.to_string(),
        "load failed: load: backend: primary; cleanup also failed: backend: cleanup"
    );
    let batch = Error::failures("close", vec![cleanup, leaf("other")]).unwrap_err();
    assert_eq!(
        batch.to_string(),
        "close encountered 2 independent failures"
    );
    assert_eq!(labels(&batch), ["primary", "cleanup", "other"]);
}

#[test]
fn stable_class_wrapper_keeps_sensitive_source_internal_and_downcastable() {
    let secret = "/private/model/shard-07 source=host.internal provider=fp8";
    for (class, expected) in [
        (ErrorClass::InvalidRequest, "invalid_request"),
        (ErrorClass::Conflict, "conflict"),
        (ErrorClass::Capacity, "capacity"),
        (ErrorClass::Unavailable, "unavailable"),
        (ErrorClass::Internal, "internal"),
    ] {
        let error = ClassifiedError::new(class, Leaf(secret));
        assert_eq!(error.class().as_str(), expected);
        assert_eq!(error.class().to_string(), expected);
        assert_eq!(
            error.source().unwrap().downcast_ref::<Leaf>().unwrap().0,
            secret
        );
        assert!(error.to_string().contains(secret)); // diagnostic text is NOT public policy
        assert_eq!(error.into_source().0, secret);
    }
}

#[test]
fn worker_arc_and_classification_wrappers_preserve_nested_sources() {
    let error = WorkerRequestError::Rejected {
        source: Arc::new(WorkerExecutionError::Runtime {
            source: ClassifiedError::new(
                ErrorClass::Internal,
                Error::with_cleanup("run", leaf("primary"), Err(leaf("cleanup"))),
            ),
        }),
    };
    assert!(
        error
            .source()
            .unwrap()
            .is::<Arc<WorkerExecutionError<ClassifiedError<Error>>>>()
    );
    let classification = ErrorChain::new(&error)
        .downcast_ref::<ClassifiedError<Error>>()
        .unwrap();
    assert_eq!(classification.class(), ErrorClass::Internal);
    assert_eq!(labels(&error), ["primary", "cleanup"]);
    assert!(
        ErrorChain::new(&error)
            .downcast_ref::<io::Error>()
            .is_none()
    );
}

#[test]
fn transparent_common_variants_are_visible_without_changing_std_source() {
    let io = Error::from(io::Error::new(
        io::ErrorKind::PermissionDenied,
        "private path",
    ));
    assert_eq!(
        ErrorChain::new(&io)
            .downcast_ref::<io::Error>()
            .unwrap()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    let protocol = Error::from(ferrule_common::IoProtocolError::EmptyDependencySet);
    assert!(
        ErrorChain::new(&protocol)
            .downcast_ref::<ferrule_common::IoProtocolError>()
            .is_some()
    );
    let resources = Error::from(
        ferrule_common::MaterializationResourceError::ZeroRequirement { resource: "slabs" },
    );
    assert!(
        ErrorChain::new(&resources)
            .downcast_ref::<ferrule_common::MaterializationResourceError>()
            .is_some()
    );
    let resolve = Error::from(
        ferrule_common::MaterializationResolveError::ProviderUnavailable {
            purpose: ferrule_common::MaterializationPurpose::Execution,
            model: ferrule_common::ModelInstanceId::new(1),
            backend: ferrule_common::BackendId::new(1),
            device: ferrule_common::DeviceId::new(0),
        },
    );
    assert!(
        ErrorChain::new(&resolve)
            .downcast_ref::<ferrule_common::MaterializationResolveError>()
            .is_some()
    );
}

#[test]
fn compatibility_exports_are_the_same_types_not_new_wrappers() {
    macro_rules! same_type {
        ($root:ty, $compat:ty, $domain:ty) => {
            assert_eq!(TypeId::of::<$root>(), TypeId::of::<$compat>());
            assert_eq!(TypeId::of::<$root>(), TypeId::of::<$domain>());
        };
    }
    same_type!(
        ferrule_common::ServingConfigError,
        ferrule_common::error::ServingConfigError,
        ferrule_common::error::serving::ServingConfigError
    );
    same_type!(
        ferrule_common::QuantizationError,
        ferrule_common::error::QuantizationError,
        ferrule_common::error::quantization::QuantizationError
    );
    same_type!(
        ferrule_common::NameMappingError,
        ferrule_common::error::NameMappingError,
        ferrule_common::error::state_dict::NameMappingError
    );
    same_type!(
        ferrule_common::IoProtocolError,
        ferrule_common::error::IoProtocolError,
        ferrule_common::error::materialization::IoProtocolError
    );
    same_type!(
        ferrule_common::WorkerRequestError<io::Error>,
        ferrule_common::error::WorkerRequestError<io::Error>,
        ferrule_common::error::serving::WorkerRequestError<io::Error>
    );
    same_type!(
        ferrule_common::StateDictBindingError<io::Error>,
        ferrule_common::error::StateDictBindingError<io::Error>,
        ferrule_common::error::state_dict::StateDictBindingError<io::Error>
    );
    same_type!(
        ferrule_common::Error,
        ferrule_common::error::Error,
        ferrule_common::error::Error
    );
    let _: ferrule_common::Result<()> = Ok(());
    let _: ferrule_common::error::Result<()> = Error::failures("empty", vec![]);
}
