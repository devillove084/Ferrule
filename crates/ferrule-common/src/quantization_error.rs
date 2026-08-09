use snafu::Snafu;

use crate::QuantType;

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum QuantizationError {
    #[snafu(display("quantization type {quant:?} is not supported by QMatrix"))]
    UnsupportedMatrixType { quant: QuantType },
    #[snafu(display("invalid {quant:?} block size: expected {expected} bytes, got {actual}"))]
    InvalidBlockSize {
        quant: QuantType,
        expected: usize,
        actual: usize,
    },
    #[snafu(display("invalid {quant:?} row value count {row_values}"))]
    InvalidRowSize { quant: QuantType, row_values: usize },
}

pub type QuantizationResult<T> = std::result::Result<T, QuantizationError>;
