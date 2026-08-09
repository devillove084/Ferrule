use std::error::Error as StdError;
use std::fmt::{Debug, Display};
use std::path::PathBuf;

use snafu::Snafu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameMappingFailureKind {
    InvalidConfiguration,
    InvalidExternalName,
    UnknownTensor,
    InvalidCanonicalPath,
    ContractViolation,
}

impl Display for NameMappingFailureKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "invalid configuration",
            Self::InvalidExternalName => "invalid external name",
            Self::UnknownTensor => "unknown tensor",
            Self::InvalidCanonicalPath => "invalid canonical path",
            Self::ContractViolation => "contract violation",
        })
    }
}

#[derive(Debug, Snafu)]
pub enum NameMappingError {
    #[snafu(display("external tensor name cannot be empty"))]
    EmptyExternalName,
    #[snafu(display("duplicate exact mapping for external tensor '{external}'"))]
    DuplicateExactMapping { external: String },
    #[snafu(display("name mapper rejected an invalid canonical path: {source}"))]
    InvalidCanonicalPath {
        source: Box<dyn StdError + Send + Sync>,
    },
    #[snafu(display("name mapper {kind}: {detail}"))]
    MapperRejected {
        kind: NameMappingFailureKind,
        detail: String,
    },
}

impl NameMappingError {
    pub fn new(reason: impl Into<String>) -> Self {
        Self::MapperRejected {
            kind: NameMappingFailureKind::ContractViolation,
            detail: reason.into(),
        }
    }

    pub fn invalid_canonical_path(source: impl StdError + Send + Sync + 'static) -> Self {
        Self::InvalidCanonicalPath {
            source: Box::new(source),
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Self::EmptyExternalName => "external tensor name cannot be empty",
            Self::DuplicateExactMapping { external } => external,
            Self::InvalidCanonicalPath { .. } => "invalid canonical path",
            Self::MapperRejected { detail, .. } => detail,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum StateDictSchemaError<Path, Id, SpecSource>
where
    Path: Debug + Display,
    Id: Debug + Display,
    SpecSource: StdError + 'static,
{
    #[snafu(display("invalid parameter '{path}': {source}"))]
    InvalidParameter { path: Path, source: SpecSource },
    #[snafu(display("duplicate canonical parameter path '{path}'"))]
    DuplicatePath { path: Path },
    #[snafu(display("duplicate parameter id {id}"))]
    DuplicateId { id: Id },
    #[snafu(display("alias '{alias}' refers to unknown parameter id {target}"))]
    UnknownAliasTarget { alias: Path, target: Id },
    #[snafu(display("parameter alias cycle detected: {cycle:?}"))]
    AliasCycle { cycle: Vec<Id> },
    #[snafu(display("alias '{alias}' is incompatible with '{target}': {reason}"))]
    IncompatibleAlias {
        alias: Path,
        target: Path,
        reason: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum StateDictTransformError {
    #[snafu(display("transpose rank mismatch: tensor rank={rank}, axes={axes:?}"))]
    TransposeRankMismatch { rank: usize, axes: Vec<usize> },
    #[snafu(display("transpose axis {axis} is outside tensor rank {rank}"))]
    TransposeAxisOutOfRange { axis: usize, rank: usize },
    #[snafu(display("transpose axis {axis} appears more than once"))]
    DuplicateTransposeAxis { axis: usize },
    #[snafu(display("split transforms are not supported by the strict binder"))]
    SplitUnsupported,
    #[snafu(display("concat transforms are not supported by the strict binder"))]
    ConcatUnsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum StateDictMetadataError {
    #[snafu(display("shape dimension {dimension} is outside the checkpoint byte-size domain"))]
    DimensionOutOfRange { dimension: usize },
    #[snafu(display("tensor element count overflowed the checkpoint byte-size domain"))]
    ElementCountOverflow,
    #[snafu(display("tensor byte size overflowed the checkpoint byte-size domain"))]
    ByteSizeOverflow,
    #[snafu(display("tensor metadata declares {actual} bytes, expected exactly {expected}"))]
    ByteSizeMismatch { expected: u64, actual: u64 },
}

#[derive(Debug, Snafu)]
pub enum StateDictBindingIssue<Path, Part, DType, NameSource, IdentitySource>
where
    Path: Debug + Display,
    Part: Debug + Display,
    DType: Debug + Display,
    NameSource: StdError + 'static,
    IdentitySource: StdError + 'static,
{
    #[snafu(display("name mapper rejected external tensor '{external}': {source}"))]
    NameMapping {
        external: String,
        source: NameSource,
    },
    #[snafu(display("unexpected external tensor '{external}'"))]
    UnexpectedTensor { external: String },
    #[snafu(display("external tensor '{external}' maps to unknown canonical path '{path}'"))]
    UnknownCanonicalPath { external: String, path: Path },
    #[snafu(display("external tensor '{external}' maps to unavailable {part} part of '{path}'"))]
    UnexpectedParameterPart {
        external: String,
        path: Path,
        part: Part,
    },
    #[snafu(display(
        "duplicate binding for {part} of '{path}': first '{first}', duplicate '{duplicate}'"
    ))]
    DuplicateTensorPart {
        path: Path,
        part: Part,
        first: String,
        duplicate: String,
    },
    #[snafu(display(
        "invalid transform for external tensor '{external}' mapped to '{path}': {source}"
    ))]
    InvalidTransform {
        external: String,
        path: Path,
        source: StateDictTransformError,
    },
    #[snafu(display(
        "dtype mismatch for external tensor '{external}' mapped to {part} of '{path}': expected {expected:?}, got {actual}"
    ))]
    DTypeMismatch {
        external: String,
        path: Path,
        part: Part,
        expected: Vec<DType>,
        actual: DType,
    },
    #[snafu(display(
        "shape mismatch for external tensor '{external}' mapped to {part} of '{path}': expected {expected:?}, got {actual:?}"
    ))]
    ShapeMismatch {
        external: String,
        path: Path,
        part: Part,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[snafu(display("invalid metadata for external tensor '{external}': {source}"))]
    InvalidByteSize {
        external: String,
        source: StateDictMetadataError,
    },
    #[snafu(display(
        "cannot capture source identity for '{external}' at '{}': {source}",
        path.display()
    ))]
    SourceIdentity {
        external: String,
        path: PathBuf,
        source: IdentitySource,
    },
    #[snafu(display(
        "source range for '{external}' at '{}' exceeds file length: offset={offset}, bytes={bytes}, source_bytes={source_bytes}",
        path.display()
    ))]
    SourceRange {
        external: String,
        path: PathBuf,
        offset: u64,
        bytes: u64,
        source_bytes: u64,
    },
    #[snafu(display("required parameter '{path}' is missing"))]
    MissingParameter { path: Path },
    #[snafu(display("required scale for parameter '{path}' is missing"))]
    MissingScale { path: Path },
    #[snafu(display("parameter '{path}' has a scale binding without a weight binding"))]
    OrphanScale { path: Path },
}

impl<Path, Part, DType, NameSource, IdentitySource>
    StateDictBindingIssue<Path, Part, DType, NameSource, IdentitySource>
where
    Path: Debug + Display,
    Part: Debug + Display,
    DType: Debug + Display,
    NameSource: StdError + 'static,
    IdentitySource: StdError + 'static,
{
    pub fn with_external(self, external: String) -> Self {
        match self {
            Self::UnexpectedParameterPart { path, part, .. } => Self::UnexpectedParameterPart {
                external,
                path,
                part,
            },
            issue => issue,
        }
    }
}

#[derive(Debug, Snafu)]
#[snafu(display(
    "state-dict binding failed with {} issue(s)",
    issues.len() + 1
))]
pub struct StateDictBindingError<Issue>
where
    Issue: StdError + Send + Sync + 'static,
{
    #[snafu(source)]
    source: Issue,
    issues: Vec<Issue>,
}

impl<Issue> StateDictBindingError<Issue>
where
    Issue: StdError + Send + Sync + 'static,
{
    pub fn new(mut issues: Vec<Issue>) -> Self {
        assert!(
            !issues.is_empty(),
            "state-dict binding errors require an issue"
        );
        let source = issues.remove(0);
        Self { source, issues }
    }

    pub fn issues(&self) -> StateDictIssues<'_, Issue> {
        StateDictIssues {
            source: &self.source,
            issues: &self.issues,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StateDictIssues<'a, Issue> {
    source: &'a Issue,
    issues: &'a [Issue],
}

impl<'a, Issue> StateDictIssues<'a, Issue> {
    pub fn iter(self) -> impl Iterator<Item = &'a Issue> {
        std::iter::once(self.source).chain(self.issues)
    }

    pub const fn len(self) -> usize {
        self.issues.len() + 1
    }

    pub const fn is_empty(self) -> bool {
        false
    }
}

impl<'a, Issue> IntoIterator for StateDictIssues<'a, Issue> {
    type Item = &'a Issue;
    type IntoIter = std::iter::Chain<std::iter::Once<&'a Issue>, std::slice::Iter<'a, Issue>>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.source).chain(self.issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping_error_preserves_typed_canonical_path_source() {
        let error = NameMappingError::invalid_canonical_path(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid module path",
        ));

        assert!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .is_some()
        );
    }

    #[test]
    fn binding_error_preserves_first_typed_issue_as_source() {
        let error = StateDictBindingError::new(vec![StateDictMetadataError::ByteSizeMismatch {
            expected: 8,
            actual: 4,
        }]);

        assert!(matches!(
            StdError::source(&error)
                .and_then(|source| source.downcast_ref::<StateDictMetadataError>()),
            Some(StateDictMetadataError::ByteSizeMismatch {
                expected: 8,
                actual: 4
            })
        ));
        assert_eq!(error.issues().len(), 1);
        assert_eq!(
            error.to_string(),
            "state-dict binding failed with 1 issue(s)"
        );
    }
}
