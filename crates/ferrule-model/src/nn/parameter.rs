use std::fmt;

use thiserror::Error;

use super::ModulePath;

/// Stable schema identity of one logical parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId(u64);

impl ParameterId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ParameterId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Storage dtype accepted by a parameter schema.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParameterDType {
    Bool,
    F16,
    F32,
    F64,
    Bf16,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Named(String),
}

impl ParameterDType {
    pub fn from_storage_name(name: &str) -> Self {
        match name {
            "BOOL" => Self::Bool,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "BF16" => Self::Bf16,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E8M0" => Self::F8E8M0,
            "I8" => Self::I8,
            "I16" => Self::I16,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "U8" => Self::U8,
            "U16" => Self::U16,
            "U32" => Self::U32,
            "U64" => Self::U64,
            other => Self::Named(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Bool => "BOOL",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::Bf16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E8M0 => "F8_E8M0",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::Named(name) => name,
        }
    }

    pub fn element_size_bytes(&self) -> Option<usize> {
        match self {
            Self::Bool | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 | Self::I8 | Self::U8 => {
                Some(1)
            }
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => Some(2),
            Self::F32 | Self::I32 | Self::U32 => Some(4),
            Self::F64 | Self::I64 | Self::U64 => Some(8),
            Self::Named(_) => None,
        }
    }

    fn validate(&self) -> Result<(), ParameterSpecError> {
        if matches!(self, Self::Named(name) if name.is_empty()) {
            Err(ParameterSpecError::EmptyDTypeName)
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for ParameterDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Non-empty set of acceptable storage dtypes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DTypeConstraint {
    allowed: Vec<ParameterDType>,
}

impl DTypeConstraint {
    pub fn exact(dtype: ParameterDType) -> Self {
        Self {
            allowed: vec![dtype],
        }
    }

    pub fn one_of(
        dtypes: impl IntoIterator<Item = ParameterDType>,
    ) -> Result<Self, ParameterSpecError> {
        let mut allowed = dtypes.into_iter().collect::<Vec<_>>();
        allowed.sort();
        allowed.dedup();
        if allowed.is_empty() {
            return Err(ParameterSpecError::EmptyDTypeConstraint);
        }
        for dtype in &allowed {
            dtype.validate()?;
        }
        Ok(Self { allowed })
    }

    pub fn accepts(&self, dtype: &ParameterDType) -> bool {
        self.allowed.binary_search(dtype).is_ok()
    }

    pub fn allowed(&self) -> &[ParameterDType] {
        &self.allowed
    }

    fn validate(&self) -> Result<(), ParameterSpecError> {
        if self.allowed.is_empty() {
            return Err(ParameterSpecError::EmptyDTypeConstraint);
        }
        for dtype in &self.allowed {
            dtype.validate()?;
        }
        if !self.allowed.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(ParameterSpecError::NonCanonicalDTypeConstraint);
        }
        Ok(())
    }
}

impl From<ParameterDType> for DTypeConstraint {
    fn from(dtype: ParameterDType) -> Self {
        Self::exact(dtype)
    }
}

/// Logical storage part of a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParameterPart {
    Weight,
    Scale,
}

impl fmt::Display for ParameterPart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Weight => "weight",
            Self::Scale => "scale",
        })
    }
}

/// Runtime placement domain without materializing the parameter payload.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParameterResidency {
    Static,
    Layer { layer: usize },
    Expert { layer: usize, expert: usize },
    Attachment { path: ModulePath },
}

impl ParameterResidency {
    pub const fn layer(layer: usize) -> Self {
        Self::Layer { layer }
    }

    pub const fn expert(layer: usize, expert: usize) -> Self {
        Self::Expert { layer, expert }
    }
}

/// Physical checkpoint encoding for one logical tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum StorageEncoding {
    /// One physical element stores one logical element.
    #[default]
    Dense,
    /// FP8 E4M3 values with an E8M0 scale for each 128x128 block.
    Fp8Block128,
    /// Two logical FP4 E2M1 values are packed into each physical I8 element.
    PackedFp4X2 { block_size: usize },
}

/// Dtype, logical shape, and physical checkpoint layout for one tensor part.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParameterTensorSpec {
    dtype: DTypeConstraint,
    logical_shape: Vec<usize>,
    physical_shape: Vec<usize>,
    encoding: StorageEncoding,
}

impl ParameterTensorSpec {
    pub fn new(
        dtype: impl Into<DTypeConstraint>,
        shape: impl Into<Vec<usize>>,
    ) -> Result<Self, ParameterSpecError> {
        let shape = shape.into();
        Self::encoded(dtype, shape.clone(), shape, StorageEncoding::Dense)
    }

    pub fn encoded(
        dtype: impl Into<DTypeConstraint>,
        logical_shape: impl Into<Vec<usize>>,
        physical_shape: impl Into<Vec<usize>>,
        encoding: StorageEncoding,
    ) -> Result<Self, ParameterSpecError> {
        let tensor = Self {
            dtype: dtype.into(),
            logical_shape: logical_shape.into(),
            physical_shape: physical_shape.into(),
            encoding,
        };
        tensor.validate()?;
        Ok(tensor)
    }

    pub fn dtype(&self) -> &DTypeConstraint {
        &self.dtype
    }

    pub fn physical_dtype(&self) -> &DTypeConstraint {
        &self.dtype
    }

    /// Compatibility alias for the logical shape.
    pub fn shape(&self) -> &[usize] {
        self.logical_shape()
    }

    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    pub const fn encoding(&self) -> StorageEncoding {
        self.encoding
    }

    fn validate(&self) -> Result<(), ParameterSpecError> {
        self.dtype.validate()?;
        validate_shape(&self.logical_shape, "logical")?;
        validate_shape(&self.physical_shape, "physical")?;
        match self.encoding {
            StorageEncoding::Dense => {
                if self.logical_shape != self.physical_shape {
                    return Err(ParameterSpecError::InvalidEncodedShape {
                        encoding: self.encoding,
                        logical: self.logical_shape.clone(),
                        physical: self.physical_shape.clone(),
                    });
                }
            }
            StorageEncoding::Fp8Block128 => {
                if self.logical_shape.len() != 2 || self.physical_shape != self.logical_shape {
                    return Err(ParameterSpecError::InvalidEncodedShape {
                        encoding: self.encoding,
                        logical: self.logical_shape.clone(),
                        physical: self.physical_shape.clone(),
                    });
                }
                ensure_exact_dtype(&self.dtype, ParameterDType::F8E4M3, self.encoding)?;
            }
            StorageEncoding::PackedFp4X2 { block_size } => {
                if block_size == 0 {
                    return Err(ParameterSpecError::ZeroEncodingBlock {
                        encoding: self.encoding,
                    });
                }
                let valid_shape = match (
                    self.logical_shape.as_slice(),
                    self.physical_shape.as_slice(),
                ) {
                    ([logical_rows, logical_cols], [physical_rows, physical_cols]) => {
                        logical_rows == physical_rows
                            && logical_cols.is_multiple_of(2)
                            && logical_cols.is_multiple_of(block_size)
                            && physical_cols.checked_mul(2) == Some(*logical_cols)
                    }
                    _ => false,
                };
                if !valid_shape {
                    return Err(ParameterSpecError::InvalidEncodedShape {
                        encoding: self.encoding,
                        logical: self.logical_shape.clone(),
                        physical: self.physical_shape.clone(),
                    });
                }
                ensure_exact_dtype(&self.dtype, ParameterDType::I8, self.encoding)?;
            }
        }
        Ok(())
    }
}

fn validate_shape(shape: &[usize], kind: &'static str) -> Result<(), ParameterSpecError> {
    if let Some((axis, _)) = shape
        .iter()
        .enumerate()
        .find(|(_, dimension)| **dimension == 0)
    {
        return Err(ParameterSpecError::ZeroShapeDimension { kind, axis });
    }
    Ok(())
}

fn ensure_exact_dtype(
    constraint: &DTypeConstraint,
    expected: ParameterDType,
    encoding: StorageEncoding,
) -> Result<(), ParameterSpecError> {
    if constraint.allowed() == [expected.clone()] {
        Ok(())
    } else {
        Err(ParameterSpecError::InvalidEncodingDType { encoding, expected })
    }
}

/// Scale-part contract for a logical parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParameterScaleSpec {
    None,
    Optional(ParameterTensorSpec),
    Required(ParameterTensorSpec),
}

impl ParameterScaleSpec {
    pub fn tensor(&self) -> Option<&ParameterTensorSpec> {
        match self {
            Self::None => None,
            Self::Optional(tensor) | Self::Required(tensor) => Some(tensor),
        }
    }

    pub const fn is_required(&self) -> bool {
        matches!(self, Self::Required(_))
    }
}

/// Immutable inference parameter declaration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParameterSpec {
    id: ParameterId,
    path: ModulePath,
    weight: ParameterTensorSpec,
    scale: ParameterScaleSpec,
    residency: ParameterResidency,
    optional: bool,
    alias_of: Option<ParameterId>,
}

impl ParameterSpec {
    pub fn new(
        id: ParameterId,
        path: ModulePath,
        dtype: impl Into<DTypeConstraint>,
        shape: impl Into<Vec<usize>>,
        residency: ParameterResidency,
    ) -> Result<Self, ParameterSpecError> {
        if path.is_root() {
            return Err(ParameterSpecError::RootParameterPath);
        }
        let parameter = Self {
            id,
            path,
            weight: ParameterTensorSpec::new(dtype, shape)?,
            scale: ParameterScaleSpec::None,
            residency,
            optional: false,
            alias_of: None,
        };
        parameter.validate()?;
        Ok(parameter)
    }

    pub fn new_encoded(
        id: ParameterId,
        path: ModulePath,
        dtype: impl Into<DTypeConstraint>,
        logical_shape: impl Into<Vec<usize>>,
        physical_shape: impl Into<Vec<usize>>,
        encoding: StorageEncoding,
        residency: ParameterResidency,
    ) -> Result<Self, ParameterSpecError> {
        if path.is_root() {
            return Err(ParameterSpecError::RootParameterPath);
        }
        let parameter = Self {
            id,
            path,
            weight: ParameterTensorSpec::encoded(dtype, logical_shape, physical_shape, encoding)?,
            scale: ParameterScaleSpec::None,
            residency,
            optional: false,
            alias_of: None,
        };
        if matches!(&parameter.residency, ParameterResidency::Attachment { path } if path.is_root())
        {
            return Err(ParameterSpecError::RootAttachmentPath);
        }
        Ok(parameter)
    }

    pub const fn id(&self) -> ParameterId {
        self.id
    }

    pub fn path(&self) -> &ModulePath {
        &self.path
    }

    pub fn dtype(&self) -> &DTypeConstraint {
        self.weight.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.weight.shape()
    }

    pub fn weight(&self) -> &ParameterTensorSpec {
        &self.weight
    }

    pub fn scale(&self) -> &ParameterScaleSpec {
        &self.scale
    }

    pub fn residency(&self) -> &ParameterResidency {
        &self.residency
    }

    pub const fn optional(&self) -> bool {
        self.optional
    }

    pub const fn alias_of(&self) -> Option<ParameterId> {
        self.alias_of
    }

    pub fn with_optional(mut self, optional: bool) -> Self {
        self.optional = optional;
        self
    }

    pub fn with_alias(mut self, target: ParameterId) -> Self {
        self.alias_of = Some(target);
        self
    }

    pub fn with_required_scale(
        mut self,
        dtype: impl Into<DTypeConstraint>,
        shape: impl Into<Vec<usize>>,
    ) -> Result<Self, ParameterSpecError> {
        self.scale = ParameterScaleSpec::Required(ParameterTensorSpec::new(dtype, shape)?);
        Ok(self)
    }

    pub fn with_optional_scale(
        mut self,
        dtype: impl Into<DTypeConstraint>,
        shape: impl Into<Vec<usize>>,
    ) -> Result<Self, ParameterSpecError> {
        self.scale = ParameterScaleSpec::Optional(ParameterTensorSpec::new(dtype, shape)?);
        Ok(self)
    }

    pub(crate) fn validate(&self) -> Result<(), ParameterSpecError> {
        if self.path.is_root() {
            return Err(ParameterSpecError::RootParameterPath);
        }
        self.weight.validate()?;
        if let Some(scale) = self.scale.tensor() {
            scale.validate()?;
        }
        self.validate_encoding_scale()?;
        if matches!(&self.residency, ParameterResidency::Attachment { path } if path.is_root()) {
            return Err(ParameterSpecError::RootAttachmentPath);
        }
        Ok(())
    }

    fn validate_encoding_scale(&self) -> Result<(), ParameterSpecError> {
        let expected_shape =
            match self.weight.encoding() {
                StorageEncoding::Dense => {
                    if self.scale.tensor().is_some()
                        && self.weight.dtype().allowed().iter().any(|dtype| {
                            matches!(dtype, ParameterDType::Bf16 | ParameterDType::F32)
                        })
                    {
                        return Err(ParameterSpecError::DenseFloatScale);
                    }
                    return Ok(());
                }
                StorageEncoding::Fp8Block128 => {
                    let [rows, cols]: [usize; 2] = self
                        .weight
                        .logical_shape()
                        .try_into()
                        .expect("FP8 tensor validation guarantees rank two");
                    vec![rows.div_ceil(128), cols.div_ceil(128)]
                }
                StorageEncoding::PackedFp4X2 { block_size } => {
                    let [rows, cols]: [usize; 2] = self
                        .weight
                        .logical_shape()
                        .try_into()
                        .expect("packed FP4 tensor validation guarantees rank two");
                    vec![rows, cols / block_size]
                }
            };
        let ParameterScaleSpec::Required(scale) = &self.scale else {
            return Err(ParameterSpecError::MissingEncodingScale {
                encoding: self.weight.encoding(),
            });
        };
        ensure_exact_dtype(
            scale.dtype(),
            ParameterDType::F8E8M0,
            self.weight.encoding(),
        )?;
        if scale.encoding() != StorageEncoding::Dense
            || scale.logical_shape() != expected_shape
            || scale.physical_shape() != expected_shape
        {
            return Err(ParameterSpecError::InvalidEncodingScale {
                encoding: self.weight.encoding(),
                expected: expected_shape,
                actual: scale.physical_shape().to_vec(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParameterSpecError {
    #[error("a parameter cannot use the root module path")]
    RootParameterPath,
    #[error("an attachment residency cannot use the root module path")]
    RootAttachmentPath,
    #[error("a dtype constraint must contain at least one dtype")]
    EmptyDTypeConstraint,
    #[error("a custom dtype name cannot be empty")]
    EmptyDTypeName,
    #[error("dtype constraints must be sorted and contain unique values")]
    NonCanonicalDTypeConstraint,
    #[error("parameter {kind} shape dimension {axis} is zero")]
    ZeroShapeDimension { kind: &'static str, axis: usize },
    #[error("{encoding:?} requires non-zero block dimensions")]
    ZeroEncodingBlock { encoding: StorageEncoding },
    #[error(
        "{encoding:?} has incompatible logical shape {logical:?} and physical shape {physical:?}"
    )]
    InvalidEncodedShape {
        encoding: StorageEncoding,
        logical: Vec<usize>,
        physical: Vec<usize>,
    },
    #[error("{encoding:?} requires exact physical dtype {expected}")]
    InvalidEncodingDType {
        encoding: StorageEncoding,
        expected: ParameterDType,
    },
    #[error("dense BF16/F32 parameters cannot declare a scale tensor")]
    DenseFloatScale,
    #[error("{encoding:?} requires a paired E8M0 scale tensor")]
    MissingEncodingScale { encoding: StorageEncoding },
    #[error("{encoding:?} scale shape must be {expected:?}, got physical shape {actual:?}")]
    InvalidEncodingScale {
        encoding: StorageEncoding,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
