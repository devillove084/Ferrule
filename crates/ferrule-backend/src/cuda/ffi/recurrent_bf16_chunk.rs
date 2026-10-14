//! Explicit workspace ABI for opt-in BF16 chunk GDN.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Bf16ChunkArgs {
    pub rows: u32,
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub reserved: u32,
    pub qkv_bf16: u64,
    pub a_bf16: u64,
    pub b_bf16: u64,
    pub a_log_bf16: u64,
    pub dt_bias_bf16: u64,
    pub state_f32: u64,
    pub output_bf16: u64,
    pub workspace_f32: u64,
    pub workspace_elements: u64,
    pub stream: u64,
}
unsafe extern "C" {
    pub(crate) fn ferrule_core_bf16_chunk_prefill_launch(args: *const Bf16ChunkArgs) -> i32;
}
const _: () = {
    assert!(size_of::<Bf16ChunkArgs>() == 104);
    assert!(std::mem::offset_of!(Bf16ChunkArgs, qkv_bf16) == 24);
    assert!(std::mem::offset_of!(Bf16ChunkArgs, workspace_f32) == 80);
    assert!(std::mem::offset_of!(Bf16ChunkArgs, stream) == 96);
};
