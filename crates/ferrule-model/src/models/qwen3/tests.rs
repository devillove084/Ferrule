use std::error::Error as _;

use crate::nn::ModulePathError;

use super::name_mapper::Qwen3HfNameMapper;

#[test]
fn name_mapper_preserves_module_path_error_source() {
    let error = Qwen3HfNameMapper::mapping("invalid-path".into())
        .expect_err("an invalid canonical path must fail");
    let source = error
        .source()
        .expect("name mapping errors must retain the canonical path source");

    assert!(source.downcast_ref::<ModulePathError>().is_some());
}
