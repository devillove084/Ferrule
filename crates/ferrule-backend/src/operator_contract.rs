//! Borrowed, provider-neutral tensor contracts for backend operators.
//!
//! These descriptors describe caller-owned views only. They contain no device
//! pointer, stream, event, allocator, or model-family state.

use crate::cuda_plan::{NumericFp8LinearPlan, NumericFp8Precision, NumericFp8SelectedKernel};
use ferrule_common::Error;
use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};
use snafu::Snafu;
use std::ops::Range;

/// Typed failures raised while validating an operator contract.
#[derive(Debug, Snafu)]
pub enum OperatorContractError {
    #[snafu(display("operator contract: {field} is invalid: {reason}"))]
    InvalidField { field: &'static str, reason: String },
    #[snafu(display("operator contract: {field} has invalid layout: {reason}"))]
    InvalidLayout { field: &'static str, reason: String },
    #[snafu(display("operator contract: {field} shape overflow"))]
    ShapeOverflow { field: &'static str },
    #[snafu(display(
        "operator contract: {field} shape mismatch: expected {expected}, got {actual}"
    ))]
    ShapeMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[snafu(display("operator contract: unsupported {capability}: {reason}"))]
    Unsupported {
        capability: &'static str,
        reason: String,
    },
    #[snafu(display(
        "operator contract: advisory alias check found {first} and {second} overlapping"
    ))]
    AdvisoryAliasing {
        first: &'static str,
        second: &'static str,
    },
}

/// Result type for descriptor construction and validation.
pub type ContractResult<T> = std::result::Result<T, OperatorContractError>;

impl From<OperatorContractError> for Error {
    fn from(source: OperatorContractError) -> Self {
        Self::Backend {
            source: Box::new(source),
        }
    }
}

fn invalid(field: &'static str, reason: impl Into<String>) -> OperatorContractError {
    OperatorContractError::InvalidField {
        field,
        reason: reason.into(),
    }
}

/// Logical element types understood by the backend contract layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalDType {
    F32,
    Bf16,
    E4M3Fn,
}

impl LogicalDType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::E4M3Fn => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Bf16 => "BF16",
            Self::E4M3Fn => "E4M3FN",
        }
    }
}

/// Caller-supplied advisory storage identity for alias diagnostics.
///
/// This is never a device address or physical proof. It can only detect an
/// overlap when callers attach the same token to two views. `None` is correct
/// when the owner has no trusted logical identity; the provider's actual
/// pointer-range check remains the final source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdvisoryStorageIdentity(u64);

impl AdvisoryStorageIdentity {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Advisory alias policy for descriptor-time diagnostics.
///
/// `Disjoint` is checked only against matching [`AdvisoryStorageIdentity`]
/// tokens. It does not replace physical provider pointer-range validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasPolicy {
    Disjoint,
    MayAlias,
}

/// Whether the descriptor may be submitted while an execution graph is captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePolicy {
    CaptureSafe,
    /// Valid for eager submission only. The existing provider performs the
    /// final capture/completion guard with `check_graph_ready`.
    EagerOnly,
}

/// Retention promised by the caller for asynchronous execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionPolicy {
    UntilCompletion,
    SubmitOnly,
}

/// A borrowed logical tensor view.
///
/// `strides_bytes` and `byte_offset` are relative to the caller's storage
/// allocation. `alignment` is the guaranteed alignment of the view start;
/// physical pointer validation remains with the existing owner/provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorViewDescriptor<'a> {
    pub dtype: LogicalDType,
    pub dims: &'a [usize],
    pub strides_bytes: &'a [usize],
    pub byte_offset: usize,
    pub alignment: usize,
    pub advisory_storage: Option<AdvisoryStorageIdentity>,
}

impl<'a> TensorViewDescriptor<'a> {
    pub const fn new(
        dtype: LogicalDType,
        dims: &'a [usize],
        strides_bytes: &'a [usize],
        byte_offset: usize,
        alignment: usize,
        advisory_storage: Option<AdvisoryStorageIdentity>,
    ) -> Self {
        Self {
            dtype,
            dims,
            strides_bytes,
            byte_offset,
            alignment,
            advisory_storage,
        }
    }

    pub fn validate(self) -> ContractResult<()> {
        if self.dims.is_empty() || self.dims.len() != self.strides_bytes.len() {
            return Err(invalid(
                "rank",
                "dimensions and strides must have the same non-zero rank",
            ));
        }
        let element_bytes = self.dtype.byte_width();
        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(invalid("alignment", "must be a non-zero power of two"));
        }
        if self.alignment < element_bytes {
            return Err(invalid(
                "alignment",
                format!(
                    "{} must be at least the {}-byte element width",
                    self.alignment, element_bytes
                ),
            ));
        }
        if !self.byte_offset.is_multiple_of(element_bytes) {
            return Err(invalid(
                "byte_offset",
                format!("must be a multiple of {element_bytes}"),
            ));
        }
        if self
            .advisory_storage
            .is_some_and(|identity| identity.value() == 0)
        {
            return Err(invalid(
                "advisory_storage",
                "identity zero is reserved for unknown storage",
            ));
        }

        let mut end = self.byte_offset;
        for (&dim, &stride) in self.dims.iter().zip(self.strides_bytes) {
            if dim == 0 {
                return Err(invalid("dims", "all dimensions must be positive"));
            }
            if stride < element_bytes || !stride.is_multiple_of(element_bytes) {
                return Err(invalid(
                    "strides_bytes",
                    format!(
                        "each stride must be an aligned byte count of at least {element_bytes}"
                    ),
                ));
            }
            let span = (dim - 1)
                .checked_mul(stride)
                .ok_or(OperatorContractError::ShapeOverflow { field: "strides" })?;
            end = end
                .checked_add(span)
                .ok_or(OperatorContractError::ShapeOverflow { field: "extent" })?;
        }
        end = end
            .checked_add(element_bytes)
            .ok_or(OperatorContractError::ShapeOverflow { field: "extent" })?;
        if end > isize::MAX as usize {
            return Err(OperatorContractError::ShapeOverflow { field: "extent" });
        }
        Ok(())
    }

    pub fn byte_range(self) -> ContractResult<Range<usize>> {
        self.validate()?;
        let mut end = self.byte_offset;
        for (&dim, &stride) in self.dims.iter().zip(self.strides_bytes) {
            end = end
                .checked_add(
                    (dim - 1)
                        .checked_mul(stride)
                        .ok_or(OperatorContractError::ShapeOverflow { field: "strides" })?,
                )
                .ok_or(OperatorContractError::ShapeOverflow { field: "extent" })?;
        }
        end = end
            .checked_add(self.dtype.byte_width())
            .ok_or(OperatorContractError::ShapeOverflow { field: "extent" })?;
        Ok(self.byte_offset..end)
    }
}

/// Scale storage plus the logical grouping geometry of a quantized weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationDescriptor<'a> {
    pub scale: TensorViewDescriptor<'a>,
    /// Logical element origin of the local weight view, not a byte address.
    pub origin: [usize; 2],
    pub group_shape: [usize; 2],
}

impl<'a> QuantizationDescriptor<'a> {
    pub const fn new(
        scale: TensorViewDescriptor<'a>,
        origin: [usize; 2],
        group_shape: [usize; 2],
    ) -> Self {
        Self {
            scale,
            origin,
            group_shape,
        }
    }

    /// Connect the scale view to Ferrule's existing numeric E4M3 layout.
    ///
    /// The scale tensor remains a generic BF16/F32 view; no fixed external
    /// scale encoding or architecture-specific ABI is introduced here.
    pub fn from_numeric_fp8_layout(
        scale: TensorViewDescriptor<'a>,
        layout: NumericFp8Layout,
    ) -> ContractResult<Self> {
        let expected_dtype = match layout.scale_type {
            NumericFp8ScaleType::Bf16 => LogicalDType::Bf16,
            NumericFp8ScaleType::F32 => LogicalDType::F32,
        };
        if scale.dtype != expected_dtype {
            return Err(invalid(
                "scale dtype",
                format!(
                    "expected {}, got {}",
                    expected_dtype.name(),
                    scale.dtype.name()
                ),
            ));
        }
        let expected_grid = numeric_scale_grid(layout)?;
        if scale.dims != expected_grid {
            return Err(OperatorContractError::ShapeMismatch {
                field: "scale grid",
                expected: format!("{:?}", expected_grid),
                actual: format!("{:?}", scale.dims),
            });
        }
        Ok(Self::new(
            scale,
            [layout.row_origin, layout.column_origin],
            [128, 128],
        ))
    }

    fn validate_for_weight(self, weight: TensorViewDescriptor<'_>) -> ContractResult<()> {
        self.scale.validate()?;
        if !matches!(self.scale.dtype, LogicalDType::Bf16 | LogicalDType::F32) {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 scales",
                reason: format!("{} scale storage is not supported", self.scale.dtype.name()),
            });
        }
        if self.group_shape.contains(&0) {
            return Err(invalid(
                "group_shape",
                "all group dimensions must be positive",
            ));
        }
        let expected_grid = [
            self.origin[0]
                .checked_rem(self.group_shape[0])
                .and_then(|offset| offset.checked_add(weight.dims[0]))
                .map(|size| size.div_ceil(self.group_shape[0]))
                .ok_or(OperatorContractError::ShapeOverflow {
                    field: "scale grid",
                })?,
            self.origin[1]
                .checked_rem(self.group_shape[1])
                .and_then(|offset| offset.checked_add(weight.dims[1]))
                .map(|size| size.div_ceil(self.group_shape[1]))
                .ok_or(OperatorContractError::ShapeOverflow {
                    field: "scale grid",
                })?,
        ];
        if self.scale.dims != expected_grid {
            return Err(OperatorContractError::ShapeMismatch {
                field: "scale grid",
                expected: format!("{:?}", expected_grid),
                actual: format!("{:?}", self.scale.dims),
            });
        }
        Ok(())
    }
}

/// Provider capability checks available for the current linear family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearCapability {
    NumericFp8,
}

/// A complete borrowed linear operator contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearOperatorDescriptor<'a> {
    pub input: TensorViewDescriptor<'a>,
    pub weight: TensorViewDescriptor<'a>,
    pub output: TensorViewDescriptor<'a>,
    pub quantization: Option<QuantizationDescriptor<'a>>,
    pub alias_policy: AliasPolicy,
    pub capture_policy: CapturePolicy,
    pub retention_policy: RetentionPolicy,
}

impl<'a> LinearOperatorDescriptor<'a> {
    pub const fn new(
        input: TensorViewDescriptor<'a>,
        weight: TensorViewDescriptor<'a>,
        output: TensorViewDescriptor<'a>,
        quantization: Option<QuantizationDescriptor<'a>>,
        alias_policy: AliasPolicy,
        capture_policy: CapturePolicy,
        retention_policy: RetentionPolicy,
    ) -> Self {
        Self {
            input,
            weight,
            output,
            quantization,
            alias_policy,
            capture_policy,
            retention_policy,
        }
    }

    pub fn validate(self) -> ContractResult<()> {
        self.input.validate()?;
        self.weight.validate()?;
        self.output.validate()?;
        if self.input.dims.len() != 2 || self.weight.dims.len() != 2 || self.output.dims.len() != 2
        {
            return Err(invalid("linear rank", "linear views must all be rank two"));
        }
        if self.input.dims[0] != self.output.dims[0]
            || self.input.dims[1] != self.weight.dims[1]
            || self.weight.dims[0] != self.output.dims[1]
        {
            return Err(OperatorContractError::ShapeMismatch {
                field: "linear dimensions",
                expected: format!(
                    "input [{}, {}], weight [{}, {}], output [{}, {}]",
                    self.input.dims[0],
                    self.input.dims[1],
                    self.weight.dims[0],
                    self.weight.dims[1],
                    self.input.dims[0],
                    self.weight.dims[0]
                ),
                actual: format!(
                    "input {:?}, weight {:?}, output {:?}",
                    self.input.dims, self.weight.dims, self.output.dims
                ),
            });
        }
        if let Some(quantization) = self.quantization {
            quantization.validate_for_weight(self.weight)?;
        }
        if self.alias_policy == AliasPolicy::Disjoint {
            check_advisory_disjoint("input", self.input, "weight", self.weight)?;
            check_advisory_disjoint("input", self.input, "output", self.output)?;
            check_advisory_disjoint("weight", self.weight, "output", self.output)?;
            if let Some(quantization) = self.quantization {
                check_advisory_disjoint("input", self.input, "scale", quantization.scale)?;
                check_advisory_disjoint("weight", self.weight, "scale", quantization.scale)?;
                check_advisory_disjoint("output", self.output, "scale", quantization.scale)?;
            }
        }
        Ok(())
    }

    pub fn validate_for(self, capability: LinearCapability) -> ContractResult<()> {
        self.validate()?;
        match capability {
            LinearCapability::NumericFp8 => self.validate_numeric_fp8(),
        }
    }

    fn validate_numeric_fp8(self) -> ContractResult<()> {
        if !matches!(self.input.dtype, LogicalDType::F32 | LogicalDType::Bf16) {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 linear input",
                reason: format!("{} input storage is not supported", self.input.dtype.name()),
            });
        }
        if self.weight.dtype != LogicalDType::E4M3Fn {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 linear weight",
                reason: format!(
                    "{} weight storage is not supported",
                    self.weight.dtype.name()
                ),
            });
        }
        if self.output.dtype != LogicalDType::F32 {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 linear output",
                reason: format!(
                    "{} output storage is not supported",
                    self.output.dtype.name()
                ),
            });
        }
        let quantization = self
            .quantization
            .ok_or(OperatorContractError::Unsupported {
                capability: "numeric FP8 quantization",
                reason: "a scale descriptor is required".into(),
            })?;
        if quantization.group_shape != [128, 128] {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 scale grouping",
                reason: "the existing numeric storage contract uses 128x128 groups".into(),
            });
        }
        if self.alias_policy != AliasPolicy::Disjoint {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 aliasing",
                reason:
                    "the existing provider requires disjoint input, weight, scale, and output views"
                        .into(),
            });
        }
        if self.retention_policy != RetentionPolicy::UntilCompletion {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 retention",
                reason: "operands must remain borrowed until the existing completion event".into(),
            });
        }
        for (name, view) in [
            ("input", self.input),
            ("weight", self.weight),
            ("output", self.output),
            ("scale", quantization.scale),
        ] {
            if view.byte_offset != 0 {
                return Err(OperatorContractError::Unsupported {
                    capability: "numeric FP8 view offsets",
                    reason: format!(
                        "{name} offsets are not expressible by the current provider ABI"
                    ),
                });
            }
            if view.dims.iter().any(|&dim| dim > i32::MAX as usize)
                || view
                    .strides_bytes
                    .iter()
                    .any(|&stride| stride / view.dtype.byte_width() > i32::MAX as usize)
            {
                return Err(OperatorContractError::Unsupported {
                    capability: "numeric FP8 dimensions",
                    reason: format!("{name} exceeds the current provider integer contract"),
                });
            }
        }
        if self.input.strides_bytes[1] != self.input.dtype.byte_width()
            || self.weight.strides_bytes[1] != self.weight.dtype.byte_width()
            || self.output.strides_bytes[1] != self.output.dtype.byte_width()
        {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 contiguous rows",
                reason: "the current provider exposes only a row stride".into(),
            });
        }
        for (name, view) in [
            ("input", self.input),
            ("weight", self.weight),
            ("output", self.output),
        ] {
            let row_bytes = view.dims[1].checked_mul(view.dtype.byte_width()).ok_or(
                OperatorContractError::ShapeOverflow {
                    field: "row extent",
                },
            )?;
            if view.strides_bytes[0] < row_bytes {
                return Err(OperatorContractError::Unsupported {
                    capability: "numeric FP8 row strides",
                    reason: format!("{name} row stride is smaller than one logical row"),
                });
            }
        }
        let scale_width = quantization.scale.dtype.byte_width();
        if quantization.scale.strides_bytes[1] != scale_width
            || quantization.scale.strides_bytes[0]
                != quantization.scale.dims[1].checked_mul(scale_width).ok_or(
                    OperatorContractError::ShapeOverflow {
                        field: "scale stride",
                    },
                )?
        {
            return Err(OperatorContractError::Unsupported {
                capability: "numeric FP8 scale layout",
                reason: "the current provider requires a contiguous row-major scale grid".into(),
            });
        }
        Ok(())
    }
}

fn numeric_scale_grid(layout: NumericFp8Layout) -> ContractResult<[usize; 2]> {
    if layout.n == 0
        || layout.n > i32::MAX as usize - 128
        || layout.k == 0
        || layout.k > i32::MAX as usize - 32
    {
        return Err(OperatorContractError::InvalidLayout {
            field: "numeric FP8 shape",
            reason: "n and k must be positive and within the provider shape limits".into(),
        });
    }
    check_layout_origin(layout.row_origin, layout.n, "row origin")?;
    check_layout_origin(layout.column_origin, layout.k, "column origin")?;
    Ok([
        (layout.row_origin % 128 + layout.n).div_ceil(128),
        (layout.column_origin % 128 + layout.k).div_ceil(128),
    ])
}

fn check_layout_origin(origin: usize, extent: usize, field: &'static str) -> ContractResult<()> {
    if origin
        .checked_add(extent)
        .is_none_or(|value| value > isize::MAX as usize)
    {
        return Err(OperatorContractError::ShapeOverflow { field });
    }
    Ok(())
}

/// Build the borrowed numeric contract without allocating or touching CUDA.
///
/// The callback runs while the descriptor borrows the local shape/stride
/// arrays. This keeps the production descriptor borrowed while allowing CPU
/// tests to inspect exactly the same construction used by both CUDA entries.
#[doc(hidden)]
pub fn build_numeric_fp8_operator_contract<R>(
    plan: NumericFp8LinearPlan,
    input_dtype: LogicalDType,
    activation_stride: usize,
    output_stride: usize,
    inspect: impl for<'a> FnOnce(LinearOperatorDescriptor<'a>) -> ContractResult<R>,
) -> ContractResult<R> {
    let layout = plan.layout();
    if plan.selected_kernel() == NumericFp8SelectedKernel::Bf16M1Gemv
        && input_dtype != LogicalDType::Bf16
    {
        return Err(OperatorContractError::Unsupported {
            capability: "numeric BF16 GEMV input",
            reason: "the M1 GEMV provider accepts only an explicit BF16 input view".into(),
        });
    }
    let scale_grid = numeric_scale_grid(layout)?;
    let input_element_bytes = input_dtype.byte_width();
    let scale_element_bytes = layout.scale_type.element_bytes();
    let input_stride_bytes = activation_stride.checked_mul(input_element_bytes).ok_or(
        OperatorContractError::ShapeOverflow {
            field: "activation stride",
        },
    )?;
    let output_stride_bytes = output_stride
        .checked_mul(LogicalDType::F32.byte_width())
        .ok_or(OperatorContractError::ShapeOverflow {
            field: "output stride",
        })?;
    let scale_row_stride = scale_grid[1].checked_mul(scale_element_bytes).ok_or(
        OperatorContractError::ShapeOverflow {
            field: "scale stride",
        },
    )?;
    let input_dims = [plan.rows(), layout.k];
    let input_strides = [input_stride_bytes, input_element_bytes];
    let weight_dims = [layout.n, layout.k];
    let weight_strides = [layout.k, 1];
    let output_dims = [plan.rows(), layout.n];
    let output_strides = [output_stride_bytes, LogicalDType::F32.byte_width()];
    let scale_strides = [scale_row_stride, scale_element_bytes];
    let input = TensorViewDescriptor::new(
        input_dtype,
        &input_dims,
        &input_strides,
        0,
        input_element_bytes,
        None,
    );
    let weight = TensorViewDescriptor::new(
        LogicalDType::E4M3Fn,
        &weight_dims,
        &weight_strides,
        0,
        1,
        None,
    );
    let output = TensorViewDescriptor::new(
        LogicalDType::F32,
        &output_dims,
        &output_strides,
        0,
        LogicalDType::F32.byte_width(),
        None,
    );
    let scale_dtype = match layout.scale_type {
        NumericFp8ScaleType::Bf16 => LogicalDType::Bf16,
        NumericFp8ScaleType::F32 => LogicalDType::F32,
    };
    let scale = TensorViewDescriptor::new(
        scale_dtype,
        &scale_grid,
        &scale_strides,
        0,
        scale_element_bytes,
        None,
    );
    let quantization = QuantizationDescriptor::from_numeric_fp8_layout(scale, layout)?;
    let capture_policy = if plan.selected_kernel() == NumericFp8SelectedKernel::Bf16M1Gemv {
        if plan.precision() != NumericFp8Precision::Bf16M1GemvF32Accumulate {
            return Err(OperatorContractError::InvalidLayout {
                field: "numeric FP8 plan",
                reason: "M1 GEMV kernel does not match its precision profile".into(),
            });
        }
        CapturePolicy::EagerOnly
    } else {
        CapturePolicy::CaptureSafe
    };
    let descriptor = LinearOperatorDescriptor::new(
        input,
        weight,
        output,
        Some(quantization),
        AliasPolicy::Disjoint,
        capture_policy,
        RetentionPolicy::UntilCompletion,
    );
    inspect(descriptor)
}

/// Advisory only: the provider's physical pointer-range check is authoritative.
fn check_advisory_disjoint(
    first_name: &'static str,
    first: TensorViewDescriptor<'_>,
    second_name: &'static str,
    second: TensorViewDescriptor<'_>,
) -> ContractResult<()> {
    let (Some(first_storage), Some(second_storage)) =
        (first.advisory_storage, second.advisory_storage)
    else {
        return Ok(());
    };
    if first_storage != second_storage {
        return Ok(());
    }
    let first_range = first.byte_range()?;
    let second_range = second.byte_range()?;
    if first_range.start < second_range.end && second_range.start < first_range.end {
        return Err(OperatorContractError::AdvisoryAliasing {
            first: first_name,
            second: second_name,
        });
    }
    Ok(())
}
