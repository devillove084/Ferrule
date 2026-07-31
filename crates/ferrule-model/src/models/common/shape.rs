//! Family-neutral shape validation for checkpoint-backed decoder components.
//!
//! Callers provide model/layer context so family adapters retain their own
//! diagnostics without duplicating the validation logic.

use crate::checkpoint::tensor::CheckpointTensorPayload;
use crate::checkpoint::weight::LinearWeight;
use ferrule_common::{Error, Result};

pub(crate) fn two_dim_shape_from_payload(
    payload: &CheckpointTensorPayload,
    label: &str,
    error_context: &str,
) -> Result<(usize, usize)> {
    let [rows, cols]: [usize; 2] =
        payload
            .slice
            .shape
            .clone()
            .try_into()
            .map_err(|shape: Vec<usize>| {
                Error::Model(format!(
                    "{error_context} {label} '{}' expects 2D shape, got {:?}",
                    payload.slice.name, shape
                ))
            })?;
    Ok((rows, cols))
}

pub(crate) fn check_linear(
    linear: &LinearWeight,
    out: usize,
    input: usize,
    label: &str,
    error_context: &str,
) -> Result<()> {
    if linear.format.out_features() != out || linear.format.in_features() != input {
        return Err(Error::Model(format!(
            "{error_context} {label} shape mismatch: got [{}, {}], expected [{out}, {input}]",
            linear.format.out_features(),
            linear.format.in_features()
        )));
    }
    Ok(())
}

pub(crate) fn check_len(
    got: usize,
    expected: usize,
    label: &str,
    error_context: &str,
) -> Result<()> {
    if got != expected {
        return Err(Error::Model(format!(
            "{error_context} {label} length mismatch: got {got}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::TensorRole;
    use crate::checkpoint::tensor::{CheckpointDType, CheckpointTensorSlice};

    #[test]
    fn extracts_two_dim_shape() {
        let payload = f32_payload("proj.weight", vec![3, 4]);
        assert_eq!(
            two_dim_shape_from_payload(&payload, "projection", "DenseDecoder").unwrap(),
            (3, 4)
        );
    }

    #[test]
    fn validators_report_caller_owned_context() {
        let vector = f32_payload("norm.weight", vec![4]);
        let error =
            two_dim_shape_from_payload(&vector, "projection", "DenseDecoder layer 2").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Model: DenseDecoder layer 2 projection 'norm.weight' expects 2D shape, got [4]"
        );

        let linear = LinearWeight::from_weight_and_scale(
            TensorRole::AttentionQuery,
            f32_payload("q_proj.weight", vec![3, 4]),
            None,
        )
        .unwrap();
        let error = check_linear(&linear, 2, 4, "q_proj", "DenseDecoder layer 2").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Model: DenseDecoder layer 2 q_proj shape mismatch: got [3, 4], expected [2, 4]"
        );

        let error = check_len(3, 4, "input_norm", "DenseDecoder layer 2").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Model: DenseDecoder layer 2 input_norm length mismatch: got 3, expected 4"
        );
    }

    fn f32_payload(name: &str, shape: Vec<usize>) -> CheckpointTensorPayload {
        let elements = shape.iter().product::<usize>();
        CheckpointTensorPayload {
            slice: CheckpointTensorSlice {
                name: name.into(),
                role: TensorRole::Unknown,
                path: PathBuf::from("synthetic.safetensors"),
                offset: 0,
                bytes: (elements * 4) as u64,
                dtype: CheckpointDType::F32,
                shape,
            },
            bytes: vec![0; elements * 4],
        }
    }
}
