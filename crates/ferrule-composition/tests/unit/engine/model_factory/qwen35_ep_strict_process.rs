use super::*;

#[cfg(unix)]
#[test]
#[ignore = "requires CUDA child, NAS FP8/reference and eight GPUs; numeric boot v2 full40"]
fn qwen35_process_ep8_factory_full40_f32_strict_oracle() {
    strict_factory(true);
}
