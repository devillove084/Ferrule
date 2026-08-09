use ferrule_common::{QuantType, QuantizationError};
use ferrule_model::quant::QMatrix;

#[test]
fn unsupported_qmatrix_type_returns_common_typed_error() {
    let error = match QMatrix::quantize(&[0.0], 1, 1, QuantType::F32) {
        Ok(_) => panic!("F32 unexpectedly quantized through QMatrix"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        QuantizationError::UnsupportedMatrixType {
            quant: QuantType::F32,
        }
    );
}
