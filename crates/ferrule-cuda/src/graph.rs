//! Feature flags for the CUDA backend (environment-controlled).

/// Whether CUDA graph capture is enabled for decode.
/// Set FERRULE_CUDA_GRAPH=1 to enable (experimental).
pub fn cuda_graph_enabled() -> bool {
    std::env::var_os("FERRULE_CUDA_GRAPH").is_some()
}

/// Whether FlashAttention-style kernels should be used.
/// Set FERRULE_FLASH_ATTN=1 to enable (requires kernel implementation).
pub fn flash_attn_enabled() -> bool {
    std::env::var_os("FERRULE_FLASH_ATTN").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_graph_enabled_defaults_false() {
        // Ensure env var is not set during test
        std::env::remove_var("FERRULE_CUDA_GRAPH");
        assert!(!cuda_graph_enabled());
    }

    #[test]
    fn cuda_graph_enabled_when_env_set() {
        std::env::set_var("FERRULE_CUDA_GRAPH", "1");
        assert!(cuda_graph_enabled());
        std::env::remove_var("FERRULE_CUDA_GRAPH");
    }
}
