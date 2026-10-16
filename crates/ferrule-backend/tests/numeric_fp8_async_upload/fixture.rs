//! Shared exact 2x2 numeric-FP8 payload crossing both 128-scale boundaries.
use ferrule_common::numeric_fp8::{
    ImmutableValidatedNumericFp8Payload, NumericFp8Layout, NumericFp8ScaleType,
};

pub(super) fn payload(
    scale_type: NumericFp8ScaleType,
    marker: u8,
) -> ImmutableValidatedNumericFp8Payload {
    let layout = NumericFp8Layout {
        n: 2,
        k: 2,
        row_origin: 127,
        column_origin: 255,
        scale_type,
    };
    let scales = match scale_type {
        NumericFp8ScaleType::Bf16 => [0x3f80u16, 0x4000, 0x4040, 0x4080]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
        NumericFp8ScaleType::F32 => [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    };
    ImmutableValidatedNumericFp8Payload::new(layout, vec![0, 0x80, marker, marker | 0x80], scales)
        .unwrap()
}
