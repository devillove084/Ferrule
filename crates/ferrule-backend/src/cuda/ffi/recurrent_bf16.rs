//! Separate, explicitly typed-storage ABI; never reinterpret the F32 entries.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Bf16ConvArgs {
    pub rows: u32,
    pub channels: u32,
    pub kernel_size: u32,
    pub reserved: u32,
    pub input_bf16: u64,
    pub weight_bf16: u64,
    pub bias_bf16: u64,
    pub history_bf16: u64,
    pub output_bf16: u64,
    pub stream: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Bf16DeltaArgs {
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
    pub stream: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Bf16GatedNormArgs {
    pub rows: u32,
    pub width: u32,
    pub epsilon: f32,
    pub reserved: u32,
    pub input_bf16: u64,
    pub weight_bf16: u64,
    pub gate_bf16: u64,
    pub output_bf16: u64,
    pub stream: u64,
}
unsafe extern "C" {
    pub(crate) fn ferrule_core_bf16_conv_launch(args: *const Bf16ConvArgs) -> i32;
    pub(crate) fn ferrule_core_bf16_delta_launch(args: *const Bf16DeltaArgs) -> i32;
    pub(crate) fn ferrule_core_bf16_gated_norm_launch(args: *const Bf16GatedNormArgs) -> i32;
}
const _: () = {
    assert!(size_of::<Bf16ConvArgs>() == 64);
    assert!(std::mem::offset_of!(Bf16ConvArgs, input_bf16) == 16);
    assert!(std::mem::offset_of!(Bf16ConvArgs, stream) == 56);
    assert!(size_of::<Bf16DeltaArgs>() == 88);
    assert!(std::mem::offset_of!(Bf16DeltaArgs, qkv_bf16) == 24);
    assert!(std::mem::offset_of!(Bf16DeltaArgs, state_f32) == 64);
    assert!(std::mem::offset_of!(Bf16DeltaArgs, stream) == 80);
    assert!(size_of::<Bf16GatedNormArgs>() == 56);
    assert!(std::mem::offset_of!(Bf16GatedNormArgs, input_bf16) == 16);
    assert!(std::mem::offset_of!(Bf16GatedNormArgs, stream) == 48);
};
