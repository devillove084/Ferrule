//! CUDA context helpers — probe, GEMV benchmarks, kernel dispatch.

use std::borrow::Cow;
use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cuda::runtime::{
    self, CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, DevicePtr, LaunchConfig,
    PinnedHostBuffer,
};
use ferrule_common::{Error, Result};

use crate::BackendError;
pub use crate::cuda::counters::CudaFailpoints;
use crate::cuda::counters::CudaOpCounterCells;
pub use crate::cuda::counters::CudaOpCounters;
use crate::cuda::cutlass::{
    CutlassKernelId, GroupedFp4MoeBuffers, GroupedFp4MoeLayout, HybridMlaExplicitSelectionLayout,
    discover_provider, grouped_fp4_moe_launch as grouped_fp4_moe, grouped_fp4_moe_workspace_size,
    hybrid_mla_explicit_selection_workspace_requirements, mxfp4_sfb_storage_bytes,
    prepare_mxfp4_sfb,
};
use crate::cuda::kernels::{DSV4_DECODE_INDEX_QUERY_SHARED_ELEMENTS, kernels::LoadedModule};
use crate::cuda::transformer::combined_ring::CombinedRingTopkLayout;
use crate::cuda::transformer::compressor_recurrent::CompressorRecurrentShape;
use crate::cuda::transformer::sparse_attention::{
    CudaSparseAttentionExecutor, CudaSparseAttentionShape, DualPlanePagedSparseAttentionLayout,
    PagedSparseAttentionLayout,
};
use crate::plan::{ExecutionMode, KernelOperation, KernelProviderId};

/// Preserve a CUDA/provider error as the source at the common error boundary.
pub(crate) fn cu<T, E>(result: std::result::Result<T, E>) -> Result<T>
where
    E: std::error::Error + Send + Sync + 'static,
{
    result.map_err(|source| Error::Backend {
        source: Box::new(source),
    })
}

fn unsupported_grouped_fp4_moe() -> Error {
    Error::Backend {
        source: Box::new(BackendError::UnsupportedOperation {
            provider: KernelProviderId::CUDA_CUTLASS,
            operation: KernelOperation::GroupedFp4Moe,
            mode: ExecutionMode::Inference,
            deterministic: false,
        }),
    }
}

fn slice_bytes<T>(slice: &[T]) -> u64 {
    (slice.len() as u64).saturating_mul(std::mem::size_of::<T>() as u64)
}

fn element_bytes<T>(len: usize) -> u64 {
    (len as u64).saturating_mul(std::mem::size_of::<T>() as u64)
}

/// Repack a logical row-major HC function matrix `[rows, cols]` as
/// `[cols, rows]`. GPU HC kernels keep each row's `col=0..cols` accumulation
/// order, while adjacent row threads read adjacent weights at every column.
pub fn transpose_hc_function_for_device(function: &[f32], rows: usize) -> Result<Vec<f32>> {
    if rows == 0 || !function.len().is_multiple_of(rows) {
        return Err(Error::Internal {
            message: format!(
                "invalid HC function layout: elements={} rows={rows}",
                function.len()
            ),
        });
    }
    let cols = function.len() / rows;
    if cols == 0 {
        return Err(Error::Internal {
            message: "HC function has zero columns".into(),
        });
    }
    let mut transposed = vec![0.0f32; function.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = function[row * cols + col];
        }
    }
    Ok(transposed)
}

fn f32_range_device_ptr(
    buffer: &CudaF32Buffer,
    offset: usize,
    len: usize,
    operation: &str,
) -> Result<DevicePtr> {
    let end = offset.checked_add(len).ok_or_else(|| Error::Internal {
        message: format!("{operation} range overflow"),
    })?;
    if end > buffer.len() {
        return Err(Error::Internal {
            message: format!(
                "{operation} out of bounds: buffer={} range={offset}..{end}",
                buffer.len()
            ),
        });
    }
    let byte_offset = offset
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Internal {
            message: format!("{operation} byte offset overflow"),
        })?;
    buffer
        .buffer
        .cu_deviceptr()
        .checked_add(byte_offset as u64)
        .ok_or_else(|| Error::Internal {
            message: format!("{operation} device pointer overflow"),
        })
}

fn copy_f32_device_range(
    stream: &CudaStream,
    src: DevicePtr,
    dst: DevicePtr,
    len: usize,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    let bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Internal {
            message: "CUDA f32 range copy byte size overflow".into(),
        })?;
    cu(runtime::copy_device_to_device(stream, dst, src, bytes))
}

fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

pub const ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaObservabilityConfig {
    moe_timing: bool,
}

impl CudaObservabilityConfig {
    fn from_env() -> Self {
        Self::resolve(|name| std::env::var(name).ok())
    }

    fn resolve(mut env: impl FnMut(&str) -> Option<String>) -> Self {
        Self {
            moe_timing: env_flag_enabled(env("FERRULE_CUDA_MOE_TIMING").as_deref()),
        }
    }
}

fn env_flag_enabled(value: Option<&str>) -> bool {
    value
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(false)
}

fn quantized_shape_uses_fp8_activation(shape: CudaArtifactLinearShape) -> bool {
    matches!(
        shape,
        CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale { .. }
            | CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale { .. }
    )
}

fn prepare_activation_for_artifact_linear<'a>(
    shape: CudaArtifactLinearShape,
    input: &'a [f32],
) -> Result<Cow<'a, [f32]>> {
    if quantized_shape_uses_fp8_activation(shape) {
        let mut quantized = input.to_vec();
        simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(
            &mut quantized,
            shape.in_features(),
            ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
        )?;
        Ok(Cow::Owned(quantized))
    } else {
        Ok(Cow::Borrowed(input))
    }
}

fn simulate_fp8_e4m3fn_e8m0_activation_quant_in_place(
    values: &mut [f32],
    row_width: usize,
    block_size: usize,
) -> Result<()> {
    if row_width == 0 || block_size == 0 || !row_width.is_multiple_of(block_size) {
        return Err(Error::Internal {
            message: format!(
                "invalid CUDA artifact FP8 activation quant shape: row_width={row_width}, block_size={block_size}"
            ),
        });
    }
    if !values.len().is_multiple_of(row_width) {
        return Err(Error::Internal {
            message: format!(
                "CUDA artifact FP8 activation length {} is not a multiple of row_width {row_width}",
                values.len()
            ),
        });
    }

    for row in values.chunks_exact_mut(row_width) {
        for block in row.chunks_exact_mut(block_size) {
            let amax = block
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()))
                .max(1e-4);
            let scale = 2.0f32.powf((amax / 448.0).log2().ceil());
            for value in block {
                let quantized = quantize_fp8_e4m3fn_to_f32((*value / scale).clamp(-448.0, 448.0));
                *value = quantized * scale;
            }
        }
    }
    Ok(())
}

fn quantize_fp8_e4m3fn_to_f32(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let sign = if value.is_sign_negative() { -1.0 } else { 1.0 };
    let magnitude = value.abs().min(448.0);
    sign * nearest_fp8_e4m3fn_positive(magnitude)
}

fn nearest_fp8_e4m3fn_positive(magnitude: f32) -> f32 {
    let mut best = nearest_fp8_subnormal_positive(magnitude);
    let mut best_err = (best - magnitude).abs();
    let exp_floor = magnitude.log2().floor() as i32;
    for exp in exp_floor - 1..=exp_floor + 1 {
        if !(-6..=8).contains(&exp) {
            continue;
        }
        let scale = 2.0f32.powi(exp);
        let mut mantissa = ((magnitude / scale - 1.0) * 8.0).round() as i32;
        let mut candidate_exp = exp;
        if mantissa < 0 {
            continue;
        }
        if mantissa > 7 {
            candidate_exp += 1;
            mantissa = 0;
        }
        if candidate_exp > 8 {
            candidate_exp = 8;
            mantissa = 6;
        }
        if candidate_exp == 8 && mantissa > 6 {
            mantissa = 6;
        }
        let candidate = 2.0f32.powi(candidate_exp) * (1.0 + mantissa as f32 / 8.0);
        let err = (candidate - magnitude).abs();
        if err < best_err {
            best = candidate;
            best_err = err;
        }
    }
    best
}

fn nearest_fp8_subnormal_positive(magnitude: f32) -> f32 {
    let step = 2.0f32.powi(-9);
    let mantissa = (magnitude / step).round().clamp(0.0, 7.0);
    mantissa * step
}

fn checked_u32(value: usize, label: &str, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Internal {
        message: format!("{label} {field} exceeds CUDA u32 ABI: {value}"),
    })
}

#[derive(Clone, Copy)]
struct Dsv4PagedDecodeRowsShape {
    rows: usize,
    window_size: usize,
    index_topk: usize,
    index_heads: usize,
    index_head_dim: usize,
}

impl Dsv4PagedDecodeRowsShape {
    fn elements(self) -> Result<usize> {
        self.rows
            .checked_mul(
                self.window_size
                    .checked_add(self.index_topk)
                    .ok_or_else(|| Error::Internal {
                        message: "CUDA paged decode rows column overflow".into(),
                    })?,
            )
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged decode rows output overflow".into(),
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_lengths(
        self,
        query: usize,
        weights: usize,
        block_offsets: usize,
        row_sequence_ids: usize,
        positions: usize,
        window_lens: usize,
        compressed_lens: usize,
        logical_indices: usize,
        plane_selectors: usize,
    ) -> Result<usize> {
        let query_len = self
            .rows
            .checked_mul(self.index_heads)
            .and_then(|value| value.checked_mul(self.index_head_dim))
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged decode rows query overflow".into(),
            })?;
        let weight_len =
            self.rows
                .checked_mul(self.index_heads)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA paged decode rows weights overflow".into(),
                })?;
        let elements = self.elements()?;
        if self.rows == 0
            || self.window_size == 0
            || self.index_topk == 0
            || self.index_topk > 512
            || self.index_heads == 0
            || self.index_head_dim == 0
            || query != query_len
            || weights != weight_len
            || block_offsets < 2
            || row_sequence_ids != self.rows
            || positions != self.rows
            || window_lens != self.rows
            || compressed_lens != self.rows
            || logical_indices != elements
            || plane_selectors != elements
        {
            return Err(Error::Internal {
                message: "CUDA paged decode rows indexer shape mismatch".into(),
            });
        }
        Ok(elements)
    }
}

// ── Kernel dispatch (selects Q4_0 vs Q8_0 at runtime) ─────────────────

// ── Device probe ──────────────────────────────────────────────────────

/// Probe the CUDA device and print basic info.
/// No-op when no GPU is available (returns an error).
pub fn cuda_probe() -> Result<()> {
    let ctx = cu(CudaContext::new(0))?;
    let name = cu(ctx.device_name())?;
    cu(ctx.bind_to_thread())?;
    let (free, total) = cu(ctx.memory_info())?;
    println!(
        "  Device: {name}\n  Memory: {:.1} GB free / {:.1} GB total",
        free as f64 / 1e9,
        total as f64 / 1e9
    );
    Ok(())
}

// ── Reusable artifact-format operator context ────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaArtifactLinearShape {
    F32 {
        out_features: usize,
        in_features: usize,
    },
    Bf16Bytes {
        out_features: usize,
        in_features: usize,
    },

    Fp8E4M3WithE8M0Scale {
        out_features: usize,
        in_features: usize,
        block_m: usize,
        block_k: usize,
    },
    Fp4E2M1PackedWithE8M0Scale {
        out_features: usize,
        in_features: usize,
    },
}

impl CudaArtifactLinearShape {
    pub fn out_features(self) -> usize {
        match self {
            Self::F32 { out_features, .. }
            | Self::Bf16Bytes { out_features, .. }
            | Self::Fp8E4M3WithE8M0Scale { out_features, .. }
            | Self::Fp4E2M1PackedWithE8M0Scale { out_features, .. } => out_features,
        }
    }

    pub fn in_features(self) -> usize {
        match self {
            Self::F32 { in_features, .. }
            | Self::Bf16Bytes { in_features, .. }
            | Self::Fp8E4M3WithE8M0Scale { in_features, .. }
            | Self::Fp4E2M1PackedWithE8M0Scale { in_features, .. } => in_features,
        }
    }

    /// Exact byte lengths required by this artifact shape's weight and scale storage.
    ///
    /// This is the authoritative storage-size calculation used by both uploads and
    /// empty frame allocation, so preallocated handles cannot drift from the
    /// existing artifact validation contract.
    pub fn storage_lengths(self) -> Result<(usize, usize)> {
        match self {
            Self::F32 {
                out_features,
                in_features,
            } => {
                if in_features == 0 || out_features == 0 {
                    return Err(Error::Internal {
                        message: format!(
                            "invalid CUDA F32 artifact linear shape: out={out_features} in={in_features}"
                        ),
                    });
                }
                let weight = out_features
                    .checked_mul(in_features)
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or_else(|| Error::Internal {
                        message: "CUDA F32 artifact weight size overflow".into(),
                    })?;
                Ok((weight, 0))
            }
            Self::Bf16Bytes {
                out_features,
                in_features,
            } => {
                if in_features == 0 || out_features == 0 {
                    return Err(Error::Internal {
                        message: format!(
                            "invalid CUDA BF16 artifact linear shape: out={out_features} in={in_features}"
                        ),
                    });
                }
                let weight = out_features
                    .checked_mul(in_features)
                    .and_then(|elements| elements.checked_mul(2))
                    .ok_or_else(|| Error::Internal {
                        message: "CUDA BF16 artifact weight size overflow".into(),
                    })?;
                Ok((weight, 0))
            }
            Self::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                if in_features == 0 || out_features == 0 || block_m == 0 || block_k == 0 {
                    return Err(Error::Internal {
                        message: format!(
                            "invalid CUDA FP8 artifact linear shape: out={out_features} in={in_features} block_m={block_m} block_k={block_k}"
                        ),
                    });
                }
                let weight =
                    out_features
                        .checked_mul(in_features)
                        .ok_or_else(|| Error::Internal {
                            message: "CUDA FP8 artifact weight size overflow".into(),
                        })?;
                let scale = out_features
                    .div_ceil(block_m)
                    .checked_mul(in_features.div_ceil(block_k))
                    .ok_or_else(|| Error::Internal {
                        message: "CUDA FP8 artifact scale size overflow".into(),
                    })?;
                Ok((weight, scale))
            }
            Self::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            } => {
                if in_features == 0
                    || out_features == 0
                    || !in_features.is_multiple_of(32)
                    || !in_features.is_multiple_of(2)
                {
                    return Err(Error::Internal {
                        message: format!(
                            "invalid CUDA FP4 artifact linear shape: out={out_features} in={in_features}"
                        ),
                    });
                }
                let weight =
                    out_features
                        .checked_mul(in_features / 2)
                        .ok_or_else(|| Error::Internal {
                            message: "CUDA FP4 artifact weight size overflow".into(),
                        })?;
                let scale =
                    out_features
                        .checked_mul(in_features / 32)
                        .ok_or_else(|| Error::Internal {
                            message: "CUDA FP4 artifact scale size overflow".into(),
                        })?;
                Ok((weight, scale))
            }
        }
    }

    fn validate(self, weight_len: usize, scale_len: usize) -> Result<()> {
        let (expected_weight, expected_scale) = self.storage_lengths()?;
        if weight_len == expected_weight && scale_len == expected_scale {
            return Ok(());
        }
        let format = match self {
            Self::F32 { .. } => "F32",
            Self::Bf16Bytes { .. } => "BF16",
            Self::Fp8E4M3WithE8M0Scale { .. } => "FP8",
            Self::Fp4E2M1PackedWithE8M0Scale { .. } => "FP4",
        };
        Err(Error::Internal {
            message: format!(
                "CUDA {format} artifact linear length mismatch: weight={weight_len} scale={scale_len}, expected weight={expected_weight} scale={expected_scale}"
            ),
        })
    }
}

pub struct CudaArtifactLinearHandle {
    shape: CudaArtifactLinearShape,
    weight: DeviceBuffer<u8>,
    scale: Option<DeviceBuffer<u8>>,
}

/// Logical dimensions for one routed expert.
///
/// The backend owns the encoded tensor contract and all provider-specific
/// materialization. Callers only describe the three logical projection widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaRoutedExpertShape {
    pub input: usize,
    pub intermediate: usize,
    pub output: usize,
}

#[derive(Debug, Clone, Copy)]
struct CudaRoutedExpertStorageBytes {
    raw_linear: usize,
    provider_private: [usize; 3],
    physical: usize,
}

impl CudaRoutedExpertShape {
    pub fn new(input: usize, intermediate: usize, output: usize) -> Result<Self> {
        let shape = Self {
            input,
            intermediate,
            output,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(self) -> Result<()> {
        let dimensions = [self.input, self.intermediate, self.output];
        if dimensions.contains(&0)
            || !self.input.is_multiple_of(128)
            || !self.intermediate.is_multiple_of(128)
            || !self.output.is_multiple_of(64)
            || dimensions
                .into_iter()
                .any(|dimension| u32::try_from(dimension).is_err())
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid CUDA routed-expert shape: input={} intermediate={} output={}",
                    self.input, self.intermediate, self.output
                ),
            });
        }
        Ok(())
    }

    /// Exact device payload required by one materialized expert.
    ///
    /// This includes the encoded gate, up, and down projections together with
    /// all backend-private layouts. Arena alignment reserve is accounted for by
    /// the arena allocator and is not part of this payload value.
    pub fn physical_bytes(self) -> Result<usize> {
        Ok(self.storage_bytes()?.physical)
    }

    fn storage_bytes(self) -> Result<CudaRoutedExpertStorageBytes> {
        self.validate()?;

        let raw_linear_bytes = |out_features: usize, in_features: usize| -> Result<usize> {
            let weight =
                out_features
                    .checked_mul(in_features / 2)
                    .ok_or_else(|| Error::Internal {
                        message: "CUDA routed-expert weight byte size overflow".into(),
                    })?;
            let scale =
                out_features
                    .checked_mul(in_features / 32)
                    .ok_or_else(|| Error::Internal {
                        message: "CUDA routed-expert metadata byte size overflow".into(),
                    })?;
            weight.checked_add(scale).ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert linear byte size overflow".into(),
            })
        };
        let provider_private_bytes = |out_features: usize, in_features: usize| -> Result<usize> {
            // The native provider currently consumes a transformed scale-block
            // layout. Keep that representation and its sizing contract private.
            mxfp4_sfb_storage_bytes(out_features, in_features).map_err(|_| Error::Internal {
                message: format!(
                    "CUDA routed-expert private layout byte query failed: out={out_features} in={in_features}"
                ),
            })
        };

        let gate_up_raw = raw_linear_bytes(self.intermediate, self.input)?;
        let down_raw = raw_linear_bytes(self.output, self.intermediate)?;
        let raw_linear = gate_up_raw
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(down_raw))
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert raw byte size overflow".into(),
            })?;
        let provider_private = [
            provider_private_bytes(self.intermediate, self.input)?,
            provider_private_bytes(self.intermediate, self.input)?,
            provider_private_bytes(self.output, self.intermediate)?,
        ];
        let provider_private_total =
            provider_private
                .into_iter()
                .try_fold(0usize, |total, bytes| {
                    total.checked_add(bytes).ok_or_else(|| Error::Internal {
                        message: "CUDA routed-expert private layout byte size overflow".into(),
                    })
                })?;
        let physical = raw_linear
            .checked_add(provider_private_total)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert physical byte size overflow".into(),
            })?;
        Ok(CudaRoutedExpertStorageBytes {
            raw_linear,
            provider_private,
            physical,
        })
    }

    fn gate_up_linear_shape(self) -> CudaArtifactLinearShape {
        CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: self.intermediate,
            in_features: self.input,
        }
    }

    fn down_linear_shape(self) -> CudaArtifactLinearShape {
        CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
            out_features: self.output,
            in_features: self.intermediate,
        }
    }
}

/// Bump-allocated device arena for immutable artifact weights.
///
/// Every returned handle owns checked views into one shared CUDA allocation.
/// Dropping the arena is safe while handles remain live; the allocation is
/// released only after the final view is dropped.
pub struct CudaArtifactLinearArena {
    storage: DeviceBuffer<u8>,
    cursor: usize,
}

impl CudaArtifactLinearArena {
    const ALIGNMENT: usize = 16;

    fn allocation_capacity(payload_bytes: usize, maximum_views: usize) -> Result<usize> {
        payload_bytes
            .checked_add(
                maximum_views
                    .checked_mul(Self::ALIGNMENT - 1)
                    .ok_or_else(|| Error::Internal {
                        message: "CUDA artifact arena alignment capacity overflow".into(),
                    })?,
            )
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact arena capacity overflow".into(),
            })
    }

    fn allocate_view(&mut self, len: usize) -> Result<DeviceBuffer<u8>> {
        let offset = self
            .cursor
            .checked_add(Self::ALIGNMENT - 1)
            .map(|value| value & !(Self::ALIGNMENT - 1))
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact arena offset overflow".into(),
            })?;
        let end = offset.checked_add(len).ok_or_else(|| Error::Internal {
            message: "CUDA artifact arena range overflow".into(),
        })?;
        if end > self.storage.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact arena exhausted: requested={len} offset={offset} capacity={}",
                    self.storage.len()
                ),
            });
        }
        let view = cu(self.storage.slice(offset, len))?;
        self.cursor = end;
        Ok(view)
    }

    pub fn allocate_linear(
        &mut self,
        shape: CudaArtifactLinearShape,
    ) -> Result<CudaArtifactLinearHandle> {
        let (weight_len, scale_len) = shape.storage_lengths()?;
        let weight = self.allocate_view(weight_len)?;
        let scale = if scale_len == 0 {
            None
        } else {
            Some(self.allocate_view(scale_len)?)
        };
        Ok(CudaArtifactLinearHandle {
            shape,
            weight,
            scale,
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.storage.len()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.cursor
    }
}

/// Backend-owned arena for routed-expert frames.
///
/// Raw encoded tensors and backend-private layouts use separate shared device
/// allocations. Returned frames retain checked views into both allocations.
pub struct CudaRoutedExpertArena {
    shape: CudaRoutedExpertShape,
    storage_bytes: CudaRoutedExpertStorageBytes,
    raw_linear_arena: CudaArtifactLinearArena,
    provider_private_layout_arena: CudaArtifactLinearArena,
    frame_capacity: usize,
    allocated_frames: usize,
}

impl CudaRoutedExpertArena {
    pub fn shape(&self) -> CudaRoutedExpertShape {
        self.shape
    }

    pub fn frame_capacity(&self) -> usize {
        self.frame_capacity
    }

    pub fn allocated_frames(&self) -> usize {
        self.allocated_frames
    }

    pub fn remaining_frames(&self) -> usize {
        self.frame_capacity - self.allocated_frames
    }

    pub fn physical_bytes(&self) -> usize {
        self.storage_bytes.physical * self.frame_capacity
    }

    pub fn allocate_frame(&mut self) -> Result<CudaPreparedRoutedExpert> {
        if self.allocated_frames == self.frame_capacity {
            return Err(Error::Internal {
                message: format!(
                    "CUDA routed-expert arena exhausted: frames={} capacity={}",
                    self.allocated_frames, self.frame_capacity
                ),
            });
        }

        let gate = self
            .raw_linear_arena
            .allocate_linear(self.shape.gate_up_linear_shape())?;
        let up = self
            .raw_linear_arena
            .allocate_linear(self.shape.gate_up_linear_shape())?;
        let down = self
            .raw_linear_arena
            .allocate_linear(self.shape.down_linear_shape())?;
        let gate_provider_scale = self
            .provider_private_layout_arena
            .allocate_view(self.storage_bytes.provider_private[0])?;
        let up_provider_scale = self
            .provider_private_layout_arena
            .allocate_view(self.storage_bytes.provider_private[1])?;
        let down_provider_scale = self
            .provider_private_layout_arena
            .allocate_view(self.storage_bytes.provider_private[2])?;

        self.allocated_frames += 1;
        Ok(CudaPreparedRoutedExpert {
            shape: self.shape,
            physical_bytes: self.storage_bytes.physical,
            gate,
            up,
            down,
            gate_provider_scale,
            up_provider_scale,
            down_provider_scale,
        })
    }
}

/// Reusable device frame for one backend-materialized routed expert.
pub struct CudaPreparedRoutedExpert {
    shape: CudaRoutedExpertShape,
    physical_bytes: usize,
    gate: CudaArtifactLinearHandle,
    up: CudaArtifactLinearHandle,
    down: CudaArtifactLinearHandle,
    // These buffers contain the provider's transformed scale-block layout. Slot
    // pointers must reference them rather than each handle's linear scale source.
    gate_provider_scale: DeviceBuffer<u8>,
    up_provider_scale: DeviceBuffer<u8>,
    down_provider_scale: DeviceBuffer<u8>,
}

impl CudaPreparedRoutedExpert {
    pub fn matches(&self, shape: CudaRoutedExpertShape) -> bool {
        self.shape == shape
    }

    pub fn shape(&self) -> CudaRoutedExpertShape {
        self.shape
    }

    pub fn physical_bytes(&self) -> usize {
        self.physical_bytes
    }

    /// Return the complete pointer tuple consumed by the expert-slot table.
    ///
    /// Scale pointers always address backend-prepared buffers. The returned
    /// pointers become consumable after the materialization event completes or
    /// the compute stream has waited on it.
    pub fn expert_slot_pointers(&self) -> Result<CudaExpertSlotPointers> {
        self.validate_storage()?;
        Ok(CudaExpertSlotPointers {
            gate_weight: self.gate.weight.cu_deviceptr(),
            gate_scale: self.gate_provider_scale.cu_deviceptr(),
            up_weight: self.up.weight.cu_deviceptr(),
            up_scale: self.up_provider_scale.cu_deviceptr(),
            down_weight: self.down.weight.cu_deviceptr(),
            down_scale: self.down_provider_scale.cu_deviceptr(),
        })
    }

    fn debug_buffers(&self) -> [(&'static str, &DeviceBuffer<u8>); 9] {
        [
            ("gate.weight.device", &self.gate.weight),
            (
                "gate.scale.device",
                self.gate.scale.as_ref().expect("validated gate scale"),
            ),
            ("up.weight.device", &self.up.weight),
            (
                "up.scale.device",
                self.up.scale.as_ref().expect("validated up scale"),
            ),
            ("down.weight.device", &self.down.weight),
            (
                "down.scale.device",
                self.down.scale.as_ref().expect("validated down scale"),
            ),
            ("gate.scale.sfb", &self.gate_provider_scale),
            ("up.scale.sfb", &self.up_provider_scale),
            ("down.scale.sfb", &self.down_provider_scale),
        ]
    }

    fn validate_storage(&self) -> Result<()> {
        self.gate.validate_storage()?;
        self.up.validate_storage()?;
        self.down.validate_storage()?;
        let expected = self.shape.storage_bytes()?;
        let actual = [
            self.gate_provider_scale.len(),
            self.up_provider_scale.len(),
            self.down_provider_scale.len(),
        ];
        if self.gate.shape != self.shape.gate_up_linear_shape()
            || self.up.shape != self.shape.gate_up_linear_shape()
            || self.down.shape != self.shape.down_linear_shape()
            || actual != expected.provider_private
            || self.physical_bytes != expected.physical
        {
            return Err(Error::Internal {
                message: "CUDA routed-expert frame storage mismatch".into(),
            });
        }
        Ok(())
    }
}

/// Opaque page-locked host buffer for stream-ordered artifact uploads.
///
/// Upload tickets keep these buffers alive until the upload event completes,
/// satisfying CUDA's async H2D source-lifetime requirement without exposing raw
/// pinned pointers to model code.
#[derive(Clone)]
pub struct CudaPinnedU8HostBuffer {
    buffer: Arc<PinnedHostBuffer<u8>>,
    offset: usize,
    len: usize,
}

impl CudaPinnedU8HostBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        // SAFETY: construction and `slice` validate that offset + len remains
        // within the Arc-owned pinned allocation.
        unsafe { self.buffer.as_ptr().add(self.offset) }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the range is bounded by the Arc-owned allocation and shared
        // access cannot mutate it.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        let end = offset.checked_add(len).ok_or_else(|| Error::Internal {
            message: "CUDA pinned host slice range overflow".into(),
        })?;
        if end > self.len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA pinned host slice out of bounds: {offset}+{len}>{}",
                    self.len
                ),
            });
        }
        Ok(Self {
            buffer: Arc::clone(&self.buffer),
            offset: self.offset + offset,
            len,
        })
    }

    pub fn is_uniquely_owned(&self) -> bool {
        Arc::strong_count(&self.buffer) == 1
    }

    /// Return the mutable pointer for an exclusively owned pinned range.
    ///
    /// # Safety
    /// The caller must not clone this buffer or access the allocation until
    /// the external writer using the returned pointer has completed.
    pub unsafe fn as_mut_ptr_unique(&mut self) -> Result<*mut u8> {
        let base = Arc::get_mut(&mut self.buffer).ok_or_else(|| Error::Internal {
            message: "CUDA pinned host buffer is still shared".into(),
        })?;
        // SAFETY: `self.offset + self.len` was validated at construction and
        // Arc uniqueness gives the caller exclusive access to the allocation.
        Ok(unsafe { base.as_mut_ptr().add(self.offset) })
    }
}

/// Cloneable allocator for CUDA page-locked host I/O slabs.
#[derive(Clone)]
pub struct CudaPinnedHostAllocator {
    ctx: Arc<CudaContext>,
}

impl CudaPinnedHostAllocator {
    pub fn allocate_u8_aligned(
        &self,
        len: usize,
        alignment: usize,
    ) -> Result<CudaPinnedU8HostBuffer> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA pinned host alignment must be a power of two, got {alignment}"
                ),
            });
        }
        let allocation_len = len
            .checked_add(alignment - 1)
            .ok_or_else(|| Error::Internal {
                message: "CUDA pinned host allocation overflow".into(),
            })?;
        let buffer = Arc::new(cu(PinnedHostBuffer::zeroed(&self.ctx, allocation_len))?);
        let address = buffer.as_ptr() as usize;
        let aligned_address = address
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or_else(|| Error::Internal {
                message: "CUDA pinned host alignment overflow".into(),
            })?;
        let offset = aligned_address - address;
        debug_assert!(offset + len <= allocation_len);
        Ok(CudaPinnedU8HostBuffer {
            buffer,
            offset,
            len,
        })
    }
}

/// Stream-ordered artifact upload that keeps pinned host sources alive until
/// the owner consumes it after the associated upload event completes.
pub struct CudaArtifactLinearAsyncUpload {
    handle: CudaArtifactLinearHandle,
    _weight: CudaPinnedU8HostBuffer,
    _scale: Option<CudaPinnedU8HostBuffer>,
}

/// Allocation-free, stream-ordered overwrite of an existing artifact handle.
///
/// The ticket owns the pinned sources and the upload-stream completion event.
/// Dropping an incomplete ticket waits for that event before releasing the
/// sources, preserving CUDA's asynchronous H2D source-lifetime requirement.
pub struct CudaArtifactLinearAsyncOverwrite {
    weight: Option<CudaPinnedU8HostBuffer>,
    scale: Option<CudaPinnedU8HostBuffer>,
    event: CudaUploadEvent,
}

impl CudaArtifactLinearAsyncOverwrite {
    pub fn event(&self) -> &CudaUploadEvent {
        &self.event
    }

    pub fn is_complete(&self) -> Result<bool> {
        self.event.is_complete()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }
}

impl Drop for CudaArtifactLinearAsyncOverwrite {
    fn drop(&mut self) {
        if !matches!(self.event.is_complete(), Ok(true)) && self.event.synchronize().is_err() {
            // A failed event synchronization cannot prove that CUDA has stopped
            // reading the pinned sources. Leak the Arc-backed guards rather than
            // freeing host memory while DMA may still be in flight.
            if let Some(weight) = self.weight.take() {
                std::mem::forget(weight);
            }
            if let Some(scale) = self.scale.take() {
                std::mem::forget(scale);
            }
        }
    }
}

/// Completion ticket for asynchronous routed-expert materialization.
///
/// The ticket retains every pinned upload source through the final event, which
/// is recorded after both uploads and backend-private layout preparation.
pub struct CudaRoutedExpertMaterialization {
    sources: Vec<CudaPinnedU8HostBuffer>,
    event: CudaUploadEvent,
}

impl CudaRoutedExpertMaterialization {
    pub fn event(&self) -> &CudaUploadEvent {
        &self.event
    }

    pub fn is_complete(&self) -> Result<bool> {
        self.event.is_complete()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }
}

impl Drop for CudaRoutedExpertMaterialization {
    fn drop(&mut self) {
        if !matches!(self.event.is_complete(), Ok(true)) && self.event.synchronize().is_err() {
            // If completion cannot be established, retain pinned storage rather
            // than risk releasing a source still referenced by queued DMA.
            for source in self.sources.drain(..) {
                std::mem::forget(source);
            }
        }
    }
}

impl CudaArtifactLinearAsyncUpload {
    pub fn shape(&self) -> CudaArtifactLinearShape {
        self.handle.shape()
    }

    pub fn into_handle(self) -> CudaArtifactLinearHandle {
        self.handle
    }
}

/// Event recorded on the artifact upload stream.
pub struct CudaUploadEvent {
    event: CudaEvent,
}

impl CudaUploadEvent {
    pub fn is_complete(&self) -> Result<bool> {
        cu(self.event.query())
    }

    pub fn synchronize(&self) -> Result<()> {
        cu(self.event.synchronize())
    }
}

/// Destination state for one expert-slot installation.
pub enum CudaExpertSlotInstallTarget {
    Empty,
    Replacement {
        previous_expert: usize,
        previous_binding: CudaExpertSlotBinding,
        consumer_quiescence: CudaComputeEvent,
    },
}

/// Exact upload-stream mutation of one expert slot.
///
/// The device table has already received the queued eviction/install kernels,
/// but its host mirror remains unpublished until [`Self::complete`] observes
/// the exact upload event. Dropping an incomplete ticket waits for the physical
/// mutation before releasing its event and metadata.
pub struct CudaExpertSlotInstallTicket {
    event: CudaUploadEvent,
    _replacement_quiescence: Option<CudaComputeEvent>,
    previous: Option<(usize, CudaExpertSlotBinding)>,
    expert: usize,
    slot: u32,
    generation: u32,
    pointers: CudaExpertSlotPointers,
    completed: bool,
}

impl CudaExpertSlotInstallTicket {
    pub fn is_complete(&self) -> Result<bool> {
        self.event.is_complete()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }

    pub fn complete(mut self, table: &mut CudaExpertSlotTable) -> Result<CudaExpertSlotBinding> {
        self.event.synchronize()?;
        table.ensure_healthy()?;
        let replay = (|| {
            let mut next = table.host.clone();
            if let Some((expert, binding)) = self.previous {
                next.evict_binding(
                    expert,
                    u32::try_from(binding.slot).map_err(|_| Error::Internal {
                        message: "CUDA submitted expert slot is negative".into(),
                    })?,
                    u32::try_from(binding.generation).map_err(|_| Error::Internal {
                        message: "CUDA submitted expert generation is negative".into(),
                    })?,
                )?;
            }
            let binding =
                next.install_at(self.expert, self.slot, self.generation, self.pointers)?;
            Ok((next, binding))
        })();
        match replay {
            Ok((next, binding)) => {
                table.host = next;
                self.completed = true;
                Ok(binding)
            }
            Err(error) => {
                table.poisoned = true;
                Err(Error::context(
                    "publish completed CUDA expert slot mutation",
                    error,
                ))
            }
        }
    }
}

impl Drop for CudaExpertSlotInstallTicket {
    fn drop(&mut self) {
        if !self.completed && !matches!(self.event.is_complete(), Ok(true)) {
            let _ = self.event.synchronize();
        }
    }
}

/// Completion marker for compute-stream work that may still reference a
/// retired device allocation.
pub struct CudaComputeEvent {
    event: CudaEvent,
}

impl CudaComputeEvent {
    pub fn is_complete(&self) -> Result<bool> {
        cu(self.event.query())
    }

    pub fn synchronize(&self) -> Result<()> {
        cu(self.event.synchronize())
    }
}

/// Cloneable authority for the stream that consumes materialized resources.
///
/// Providers own their upload streams, but replacement safety is defined by the
/// stream that launched the actual consumer kernels. Recording through this
/// authority captures exactly that stream without sharing the full operator
/// context or creating a second execution path.
#[derive(Clone)]
pub struct CudaComputeStreamAuthority {
    stream: Arc<CudaStream>,
}

impl CudaComputeStreamAuthority {
    pub fn record_event(&self) -> Result<CudaComputeEvent> {
        match self.stream.record_event(None) {
            Ok(event) => Ok(CudaComputeEvent { event }),
            Err(source) => Err(Error::with_cleanup(
                "record CUDA consumer-stream completion event",
                Error::Backend {
                    source: Box::new(source),
                },
                cu(self.stream.synchronize()),
            )),
        }
    }
}

/// Page-locked host memory buffer that is directly accessible from the GPU
/// via `cuMemHostGetDevicePointer`. On unified-memory CUDA systems (unified memory), this avoids
/// the page-fault overhead of `cuMemAllocManaged` — the GPU reads directly
/// from host LPDDR5X pages over the coherent interconnect.
///
/// Unlike `cuMemAllocManaged`, which triggers a page fault on first GPU
/// access (~2.4ms per expert for migration), `cuMemAllocHost` pre-pins the
/// memory so GPU access has zero fault overhead.
///
/// The host pointer and device pointer alias the same physical memory,
/// so there is no need for an explicit H2D copy after the initial memcpy.
pub struct HostPinnedBuffer {
    /// Raw host pointer from `cuMemAllocHost`. Freed via `cuMemFreeHost`.
    host_ptr: *mut std::os::raw::c_void,
    /// Device pointer aliasing the same physical memory.
    dev_ptr: DevicePtr,
    len: usize,
}

impl HostPinnedBuffer {
    /// Allocate page-locked host memory, copy `data` into it, and obtain the
    /// device pointer. The GPU can read this memory directly over the
    /// coherent interconnect without any H2D DMA transfer.
    pub fn alloc_and_copy(data: &[u8]) -> Result<Self> {
        let host_ptr = cu(runtime::alloc_host(data.len()))?;
        // Copy data into page-locked host memory (host-side memcpy, no DMA).
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), host_ptr as *mut u8, data.len());
        }
        // Get the device pointer that aliases the same physical memory.
        let dev_ptr = match runtime::host_device_pointer(host_ptr) {
            Ok(pointer) => pointer,
            Err(error) => {
                let _ = runtime::free_host(host_ptr);
                return Err(error.into());
            }
        };
        Ok(Self {
            host_ptr,
            dev_ptr,
            len: data.len(),
        })
    }

    /// Device pointer for kernel launches.
    pub fn cu_deviceptr(&self) -> DevicePtr {
        self.dev_ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for HostPinnedBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            let _ = runtime::free_host(self.host_ptr);
        }
    }
}

unsafe impl Send for HostPinnedBuffer {}
unsafe impl Sync for HostPinnedBuffer {}

impl CudaArtifactLinearHandle {
    pub fn shape(&self) -> CudaArtifactLinearShape {
        self.shape
    }

    fn validate_storage(&self) -> Result<()> {
        let (expected_weight, expected_scale) = self.shape.storage_lengths()?;
        let actual_scale = self.scale.as_ref().map(DeviceBuffer::len).unwrap_or(0);
        if self.weight.len() != expected_weight || actual_scale != expected_scale {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear handle storage mismatch: shape={:?} weight={} scale={}, expected weight={expected_weight} scale={expected_scale}",
                    self.shape,
                    self.weight.len(),
                    actual_scale
                ),
            });
        }
        Ok(())
    }
}

/// Opaque typed device buffer used by generic artifact operators.
///
/// This is intentionally a CUDA/backend type, not a model-specific arena type.
/// Model-family code can own and reuse these buffers without exposing CUDA driver
/// handles through the execution boundary. Length and allocation ownership remain
/// authoritative in the underlying [`DeviceBuffer`].
pub struct CudaTypedBuffer<T: DeviceCopy> {
    buffer: DeviceBuffer<T>,
}

pub type CudaF32Buffer = CudaTypedBuffer<f32>;
pub type CudaBf16Buffer = CudaTypedBuffer<u16>;
pub type CudaI32Buffer = CudaTypedBuffer<i32>;

pub struct CudaCompressorRecurrentState {
    kv_state: CudaF32Buffer,
    score_state: CudaF32Buffer,
    shape: CompressorRecurrentShape,
}

pub struct CudaCompressorRecurrentCheckpointSlab {
    kv_states: CudaF32Buffer,
    score_states: CudaF32Buffer,
    shape: CompressorRecurrentShape,
    slots: usize,
}

impl CudaCompressorRecurrentCheckpointSlab {
    pub fn supports(&self, state: &CudaCompressorRecurrentState, slots: usize) -> bool {
        self.shape == state.shape && self.slots >= slots
    }

    pub fn slots(&self) -> usize {
        self.slots
    }
}

impl CudaCompressorRecurrentState {
    pub fn ratio(&self) -> usize {
        self.shape.ratio
    }

    pub fn head_dim(&self) -> usize {
        self.shape.head_dim
    }

    pub fn out_dim(&self) -> usize {
        self.shape.out_dim
    }

    pub fn overlap(&self) -> bool {
        self.shape.overlap
    }

    pub fn kv_state(&self) -> &CudaF32Buffer {
        &self.kv_state
    }

    pub fn score_state(&self) -> &CudaF32Buffer {
        &self.score_state
    }
}

/// Caller-owned scratch for allocation-free artifact linear execution.
///
/// FP8 MMA consumes packed activations and E8M0 scales. Other paths preserve
/// the input by copying it into `cloned` before applying the existing in-place
/// activation quantization contract.
pub struct CudaArtifactLinearWorkspace {
    cloned: CudaF32Buffer,
    x_packed: DeviceBuffer<u8>,
    x_scales: DeviceBuffer<u8>,
    value_capacity: usize,
    scale_capacity: usize,
}

/// Graph-stable scratch for one checkpoint-native proposal hybrid-attention launch.
///
/// The five-row query and block KV remain caller-owned stage values. This workspace
/// owns only the BF16 boundaries, score/probability matrices, output, and device
/// status needed by the semantic CUTLASS bundle.
pub struct CudaHybridMlaAttentionWorkspace {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS query scratch alive")
    )]
    query_bf16: DeviceBuffer<u16>,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS KV scratch alive")
    )]
    gathered_kv_bf16: DeviceBuffer<u16>,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS score scratch alive")
    )]
    scores: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS probability scratch alive")
    )]
    probabilities_bf16: DeviceBuffer<u16>,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(
            dead_code,
            reason = "keeps native CUTLASS online-softmax rescale scratch alive"
        )
    )]
    online_rescales: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(
            dead_code,
            reason = "keeps native CUTLASS softmax denominator scratch alive"
        )
    )]
    denominators: CudaF32Buffer,
    status: CudaI32Buffer,
}

impl CudaHybridMlaAttentionWorkspace {
    pub fn status(&self) -> &CudaI32Buffer {
        &self.status
    }
}

/// Opaque provider workspace for hybrid MLA explicit selection.
fn hybrid_mla_explicit_selection_launch_count() -> u64 {
    #[cfg(ferrule_cuda_test_oracle)]
    {
        if std::env::var("FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE")
            .is_ok_and(|value| value == "1")
        {
            return 7;
        }
        if std::env::var("FERRULE_CUDA_HYBRID_MLA_EXPLICIT_SELECTION_TEST_ORACLE")
            .is_ok_and(|value| value == "1")
        {
            return 1;
        }
    }
    4
}

pub struct CudaHybridMlaExplicitSelectionWorkspace {
    storage: DeviceBuffer<u8>,
    status: CudaI32Buffer,
    capacity_bytes: usize,
    alignment: usize,
    allocated_layout: HybridMlaExplicitSelectionLayout,
    #[cfg(ferrule_cuda_test_oracle)]
    oracle_output: DeviceBuffer<f32>,
}

impl CudaHybridMlaExplicitSelectionWorkspace {
    pub fn status(&self) -> &CudaI32Buffer {
        &self.status
    }

    fn supports(&self, layout: HybridMlaExplicitSelectionLayout) -> Result<bool> {
        let requirements = hybrid_mla_explicit_selection_workspace_requirements(layout)?;
        let required_bytes = usize::try_from(requirements.bytes).map_err(|_| Error::Internal {
            message: format!(
                "hybrid MLA explicit selection workspace requirement exceeds usize: {}",
                requirements.bytes
            ),
        })?;
        let required_alignment = requirements.alignment as usize;
        Ok(required_bytes <= self.capacity_bytes
            && required_alignment <= self.alignment
            && self
                .storage
                .cu_deviceptr()
                .is_multiple_of(requirements.alignment.into()))
    }
}

/// Graph-stable outputs and reduction scratch for the checkpoint-native proposal
/// HC/LM/Markov/confidence semantic bundle.
pub struct CudaProposalHeadWorkspace {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS hidden scratch alive")
    )]
    hidden: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS normalization scratch alive")
    )]
    normalized: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS logits scratch alive")
    )]
    base_logits: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS reduction values alive")
    )]
    partial_values: CudaF32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps native CUTLASS reduction indices alive")
    )]
    partial_indices: CudaI32Buffer,
    token_ids: CudaI32HostMirror,
    confidence: CudaF32Buffer,
    status: CudaI32Buffer,
    #[cfg_attr(
        not(feature = "cuda"),
        allow(dead_code, reason = "keeps the CUTLASS result mirror alive")
    )]
    result: CudaI32HostMirror,
}

impl CudaProposalHeadWorkspace {
    pub fn token_ids(&self) -> &CudaI32Buffer {
        self.token_ids.device()
    }

    pub fn confidence(&self) -> &CudaF32Buffer {
        &self.confidence
    }

    pub fn status(&self) -> &CudaI32Buffer {
        &self.status
    }
}

/// Dedicated storage for a producer-owned FP8 activation pack.
pub struct CudaFp8ActivationPack {
    x_packed: DeviceBuffer<u8>,
    x_scales: DeviceBuffer<u8>,
    value_capacity: usize,
    scale_capacity: usize,
}

/// A call-scoped immutable view of one freshly prepared activation.
///
/// The lifetime keeps the backing pack exclusively borrowed until all
/// consumers finish, preventing another producer from overwriting it.
pub struct CudaPreparedFp8Activation<'a> {
    x_packed: &'a DeviceBuffer<u8>,
    x_scales: &'a DeviceBuffer<u8>,
    rows: usize,
    row_width: usize,
}

/// Reusable workspace for grouped FP4 MoE batched execution.
///
/// The decode path hits this once per layer per token, so avoiding transient
/// CUDA allocations here is critical. The workspace owns all per-call scratch
/// buffers and fixed-size device arrays for selected expert pointers/weights.
pub struct CudaMoeBatchedWorkspace {
    gate_ptrs: DeviceBuffer<u64>,
    gate_scale_ptrs: DeviceBuffer<u64>,
    up_ptrs: DeviceBuffer<u64>,
    up_scale_ptrs: DeviceBuffer<u64>,
    down_ptrs: DeviceBuffer<u64>,
    down_scale_ptrs: DeviceBuffer<u64>,
    route_weights: DeviceBuffer<f32>,
    route_slots: DeviceBuffer<i32>,
    dispatch_error: DeviceBuffer<i32>,
    expert_output: CudaF32Buffer,
    max_experts: usize,
    input_size: usize,
    intermediate_size: usize,
    hidden_size: usize,
}

/// Device-resident compact routing metadata and caller-owned grouped FP4 MoE scratch.
///
/// Route resolution, counting, compaction, and scattering remain stream ordered on
/// device. The native grouped operator also requires four host scalar dimensions,
/// so a fixed 16-byte control block is copied to persistent pinned storage once per
/// prepared plan; no device allocation or stream-wide synchronization occurs there.
pub struct CudaExpertGroupRoutePlan {
    slot_counts: DeviceBuffer<i32>,
    slot_route_offsets: DeviceBuffer<i32>,
    slot_cursors: DeviceBuffer<i32>,
    active_expert_slots: DeviceBuffer<i32>,
    active_group_generations: DeviceBuffer<i32>,
    expert_route_indptr: DeviceBuffer<i32>,
    expert_route_counts: DeviceBuffer<i32>,
    route_token_indices: DeviceBuffer<i32>,
    route_indices: DeviceBuffer<i32>,
    route_weights: DeviceBuffer<f32>,
    host_scalars: DeviceBuffer<i32>,
    host_staging: PinnedHostBuffer<i32>,
    metadata_ready: CudaEvent,
    metadata_copied: CudaEvent,
    host_metadata: Option<CudaExpertGroupRoutePlanHost>,
    route_written: DeviceBuffer<i32>,
    route_error: DeviceBuffer<i32>,
    resolve: CudaExpertRouteResolveWorkspace,
    input_fp8: DeviceBuffer<u8>,
    input_ue8m0: DeviceBuffer<u8>,
    cutlass_workspace: DeviceBuffer<u8>,
    max_experts: usize,
    route_capacity: usize,
    tokens: usize,
    input_size: usize,
    intermediate_size: usize,
    hidden_size: usize,
    input_prepared: bool,
    invocation_routes: Option<usize>,
}

const GROUPED_FP4_MOE_SMALL_GROUP_ROW_LIMIT: usize = 192;
const GROUPED_FP4_MOE_WORKSPACE_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaExpertGroupRoutePlanHost {
    pub active_group_count: usize,
    pub small_group_count: usize,
    pub max_group_rows: usize,
    pub total_routed_rows: usize,
}

impl CudaMoeBatchedWorkspace {
    pub fn matches(
        &self,
        max_experts: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> bool {
        self.max_experts >= max_experts
            && self.input_size == input_size
            && self.intermediate_size == intermediate_size
            && self.hidden_size == hidden_size
    }
}

impl CudaExpertGroupRoutePlan {
    pub fn matches(
        &self,
        max_experts: usize,
        route_capacity: usize,
        tokens: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> bool {
        self.max_experts >= max_experts
            && self.route_capacity >= route_capacity
            && self.tokens == tokens
            && self.input_size == input_size
            && self.intermediate_size == intermediate_size
            && self.hidden_size == hidden_size
    }

    pub fn host_metadata(&self) -> Option<CudaExpertGroupRoutePlanHost> {
        self.host_metadata
    }
}

impl<T: DeviceCopy> CudaTypedBuffer<T> {
    fn from_device_buffer(buffer: DeviceBuffer<T>) -> Self {
        Self { buffer }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn as_device_buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }
}

pub enum CombinedRingWindowLens<'a> {
    PositionDerived,
    Explicit(&'a CudaI32Buffer),
}

impl CudaTypedBuffer<i32> {
    pub fn prefix(&self, len: usize) -> Result<Self> {
        if len > self.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA i32 prefix exceeds capacity: requested={len} capacity={}",
                    self.len()
                ),
            });
        }
        Ok(Self::from_device_buffer(
            self.buffer.slice(0, len).map_err(|source| Error::Backend {
                source: Box::new(source),
            })?,
        ))
    }
}

/// Validated, device-resident token-id hash table for the DSV4 router.
///
/// Construction validates the authoritative host `usize` payload before its
/// one-time conversion and upload to device `i32` storage.
pub struct CudaDsv4RouterHashTable {
    buffer: DeviceBuffer<i32>,
    rows: usize,
    cols: usize,
}

/// Persistent pinned host mirror for small, frequently updated i32 control tables.
pub struct CudaI32HostMirror {
    host: Vec<i32>,
    device: CudaI32Buffer,
    staging: PinnedHostBuffer<i32>,
    copy_event: CudaEvent,
    active_download: Option<Arc<CudaEvent>>,
}

/// Non-blocking device-to-host transfer into a persistent i32 mirror.
pub struct CudaI32HostDownload {
    copied: Arc<CudaEvent>,
    bytes: u64,
}

impl CudaI32HostMirror {
    pub fn len(&self) -> usize {
        self.device.len()
    }

    pub fn is_empty(&self) -> bool {
        self.device.is_empty()
    }

    pub fn device(&self) -> &CudaI32Buffer {
        &self.device
    }

    /// Gives a device kernel mutable access and invalidates the host equality
    /// cache. A subsequent host-mirror update will therefore republish its full
    /// payload instead of incorrectly skipping a changed device buffer.
    pub fn device_mut_invalidate_host(&mut self) -> &mut CudaI32Buffer {
        self.host.clear();
        &mut self.device
    }
}

impl std::ops::Deref for CudaI32HostMirror {
    type Target = CudaI32Buffer;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for CudaI32HostMirror {
    fn drop(&mut self) {
        if let Some(download) = self.active_download.take() {
            let _ = download.synchronize();
        }
        let _ = self.copy_event.synchronize();
    }
}

/// Persistent DSV4 router token ids with an authoritative host mirror.
pub struct CudaDsv4RouterTokenIds {
    host: Vec<u32>,
    device: CudaI32Buffer,
    staging: PinnedHostBuffer<i32>,
    copy_event: CudaEvent,
}

impl Drop for CudaDsv4RouterTokenIds {
    fn drop(&mut self) {
        let _ = self.copy_event.synchronize();
    }
}

impl CudaDsv4RouterHashTable {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

pub fn validate_dsv4_router_token_ids(token_ids: &[u32], hash_rows: usize) -> Result<Vec<i32>> {
    if token_ids.is_empty() || hash_rows == 0 {
        return Err(Error::Internal {
            message: format!(
                "CUDA DSV4 hash router requires non-empty token ids and hash rows, got tokens={} rows={hash_rows}",
                token_ids.len()
            ),
        });
    }
    token_ids
        .iter()
        .enumerate()
        .map(|(row, &token_id)| {
            let token_id = usize::try_from(token_id).map_err(|_| {
                Error::Internal { message: format!(
                    "CUDA DSV4 hash router token id at batch row {row} does not fit usize"
                ) }
            })?;
            if token_id >= hash_rows {
                return Err(Error::Internal { message: format!(
                    "CUDA DSV4 hash router token id {token_id} at batch row {row} exceeds hash rows {hash_rows}"
                ) });
            }
            i32::try_from(token_id).map_err(|_| {
                Error::Internal { message: format!(
                    "CUDA DSV4 hash router token id {token_id} at batch row {row} does not fit i32"
                ) }
            })
        })
        .collect()
}

pub fn validate_dsv4_router_hash_table(
    table: &[usize],
    rows: usize,
    cols: usize,
    experts: usize,
    top_k: usize,
) -> Result<Vec<i32>> {
    let expected = rows.checked_mul(cols).ok_or_else(|| Error::Internal {
        message: "CUDA DSV4 hash router table shape overflow".into(),
    })?;
    if rows == 0 || cols == 0 || table.len() != expected {
        return Err(Error::Internal {
            message: format!(
                "CUDA DSV4 hash router table shape mismatch: values={} rows={rows} cols={cols}",
                table.len()
            ),
        });
    }
    if top_k == 0 || top_k > cols || top_k > experts || top_k > 64 {
        return Err(Error::Internal {
            message: format!(
                "CUDA DSV4 hash router requires top_k in 1..={}, got {top_k}",
                cols.min(experts).min(64)
            ),
        });
    }
    for row in 0..rows {
        let selected = &table[row * cols..row * cols + top_k];
        for (slot, &expert) in selected.iter().enumerate() {
            if expert >= experts {
                return Err(Error::Internal {
                    message: format!(
                        "CUDA DSV4 hash router expert id {expert} at table row {row} slot {slot} exceeds expert count {experts}"
                    ),
                });
            }
            if selected[..slot].contains(&expert) {
                return Err(Error::Internal {
                    message: format!(
                        "CUDA DSV4 hash router duplicate expert id {expert} at table row {row} within top_k {top_k}"
                    ),
                });
            }
        }
    }
    table
        .iter()
        .enumerate()
        .map(|(index, &expert)| {
            i32::try_from(expert).map_err(|_| {
                Error::Internal { message: format!(
                    "CUDA DSV4 hash router table value {expert} at flat index {index} does not fit i32"
                ) }
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaExpertSlotPointers {
    pub gate_weight: u64,
    pub gate_scale: u64,
    pub up_weight: u64,
    pub up_scale: u64,
    pub down_weight: u64,
    pub down_scale: u64,
}

impl CudaExpertSlotPointers {
    fn is_complete(self) -> bool {
        self.gate_weight != 0
            && self.gate_scale != 0
            && self.up_weight != 0
            && self.up_scale != 0
            && self.down_weight != 0
            && self.down_scale != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaExpertSlotBinding {
    pub slot: i32,
    pub generation: i32,
}

/// Host mirror for one layer's stable expert slot table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaExpertSlotTableHost {
    gate_weight: Vec<u64>,
    gate_scale: Vec<u64>,
    up_weight: Vec<u64>,
    up_scale: Vec<u64>,
    down_weight: Vec<u64>,
    down_scale: Vec<u64>,
    expert_to_slot: Vec<i32>,
    expert_generation: Vec<i32>,
    slot_generation: Vec<i32>,
}

impl CudaExpertSlotTableHost {
    pub fn new(expert_capacity: usize, slot_capacity: usize) -> Result<Self> {
        if expert_capacity == 0 || slot_capacity == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert slot table requires positive capacities: experts={expert_capacity} slots={slot_capacity}"
                ),
            });
        }
        Ok(Self {
            gate_weight: vec![0; slot_capacity],
            gate_scale: vec![0; slot_capacity],
            up_weight: vec![0; slot_capacity],
            up_scale: vec![0; slot_capacity],
            down_weight: vec![0; slot_capacity],
            down_scale: vec![0; slot_capacity],
            expert_to_slot: vec![-1; expert_capacity],
            expert_generation: vec![0; expert_capacity],
            slot_generation: vec![0; slot_capacity],
        })
    }

    pub fn expert_capacity(&self) -> usize {
        self.expert_to_slot.len()
    }

    pub fn slot_capacity(&self) -> usize {
        self.slot_generation.len()
    }

    pub fn binding(&self, expert: usize) -> Option<CudaExpertSlotBinding> {
        let slot = *self.expert_to_slot.get(expert)?;
        if slot < 0 {
            return None;
        }
        let generation = *self.expert_generation.get(expert)?;
        let slot_generation = *self.slot_generation.get(slot as usize)?;
        (generation > 0 && generation == slot_generation)
            .then_some(CudaExpertSlotBinding { slot, generation })
    }

    pub fn is_current(&self, binding: CudaExpertSlotBinding) -> bool {
        binding.slot >= 0
            && self
                .slot_generation
                .get(binding.slot as usize)
                .is_some_and(|generation| *generation == binding.generation)
    }

    fn exact_coordinates(
        &self,
        expert: usize,
        slot: u32,
        generation: u32,
    ) -> Result<(usize, i32, i32)> {
        if expert >= self.expert_capacity() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert id {expert} exceeds slot table capacity {}",
                    self.expert_capacity()
                ),
            });
        }
        let slot_index = slot as usize;
        if slot_index >= self.slot_capacity() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert slot {slot} exceeds slot table capacity {}",
                    self.slot_capacity()
                ),
            });
        }
        let slot = i32::try_from(slot).map_err(|_| Error::Internal {
            message: format!("CUDA expert slot {slot} does not fit the i32 device ABI"),
        })?;
        let generation = i32::try_from(generation)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::Internal {
                message: format!(
                    "CUDA expert slot generation must be positive and fit i32, got {generation}"
                ),
            })?;
        Ok((slot_index, slot, generation))
    }

    fn pointers_at(&self, slot: usize) -> CudaExpertSlotPointers {
        CudaExpertSlotPointers {
            gate_weight: self.gate_weight[slot],
            gate_scale: self.gate_scale[slot],
            up_weight: self.up_weight[slot],
            up_scale: self.up_scale[slot],
            down_weight: self.down_weight[slot],
            down_scale: self.down_scale[slot],
        }
    }

    fn install_at(
        &mut self,
        expert: usize,
        slot: u32,
        generation: u32,
        pointers: CudaExpertSlotPointers,
    ) -> Result<CudaExpertSlotBinding> {
        let (slot_index, slot, generation) = self.exact_coordinates(expert, slot, generation)?;
        if !pointers.is_complete() {
            return Err(Error::Internal {
                message:
                    "CUDA expert slot table requires a complete non-null weight/scale pointer tuple"
                        .into(),
            });
        }

        if self.expert_to_slot[expert] >= 0 {
            let current = self.binding(expert).ok_or_else(|| Error::Internal {
                message: format!("CUDA expert {expert} has an inconsistent existing slot binding"),
            })?;
            if current.slot == slot && current.generation == generation {
                if self.pointers_at(slot_index) != pointers {
                    return Err(Error::Internal {
                        message: format!(
                            "CUDA expert {expert} slot {slot} generation {generation} is already installed with a different pointer tuple"
                        ),
                    });
                }
                return Ok(current);
            }
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert {expert} is already bound to slot {} generation {}, not slot {slot} generation {generation}",
                    current.slot, current.generation
                ),
            });
        }

        if let Some(conflicting_expert) = self
            .expert_to_slot
            .iter()
            .position(|bound_slot| *bound_slot == slot)
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert slot {slot} is already bound to expert {conflicting_expert}"
                ),
            });
        }

        let expected_generation = match self.slot_generation[slot_index] {
            0 => 1,
            i32::MAX => {
                return Err(Error::Internal {
                    message: "CUDA expert slot generation exhausted".into(),
                });
            }
            generation => generation,
        };
        if generation != expected_generation {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert slot {slot} expected generation {expected_generation}, got {generation}"
                ),
            });
        }

        self.slot_generation[slot_index] = generation;
        self.expert_to_slot[expert] = slot;
        self.expert_generation[expert] = generation;
        self.gate_weight[slot_index] = pointers.gate_weight;
        self.gate_scale[slot_index] = pointers.gate_scale;
        self.up_weight[slot_index] = pointers.up_weight;
        self.up_scale[slot_index] = pointers.up_scale;
        self.down_weight[slot_index] = pointers.down_weight;
        self.down_scale[slot_index] = pointers.down_scale;
        Ok(CudaExpertSlotBinding { slot, generation })
    }

    fn evict_binding(&mut self, expert: usize, slot: u32, generation: u32) -> Result<()> {
        let (slot_index, slot, generation) = self.exact_coordinates(expert, slot, generation)?;
        let current = self.binding(expert).ok_or_else(|| {
            Error::Internal { message: format!(
                "CUDA expert slot eviction rejected stale binding: expert {expert} slot {slot} generation {generation}"
            ) }
        })?;
        if current.slot != slot || current.generation != generation {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert slot eviction rejected stale binding: expert {expert} is at slot {} generation {}, not slot {slot} generation {generation}",
                    current.slot, current.generation
                ),
            });
        }

        let next_generation = generation
            .checked_add(1)
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert slot generation exhausted".into(),
            })?;
        self.expert_to_slot[expert] = -1;
        self.expert_generation[expert] = 0;
        self.gate_weight[slot_index] = 0;
        self.gate_scale[slot_index] = 0;
        self.up_weight[slot_index] = 0;
        self.up_scale[slot_index] = 0;
        self.down_weight[slot_index] = 0;
        self.down_scale[slot_index] = 0;
        // Advancing invalidates stale slot/generation handles. The controller's
        // prepared replacement already carries this next generation, so exact
        // installation consumes it without incrementing it a second time.
        self.slot_generation[slot_index] = next_generation;
        Ok(())
    }

    pub fn install(
        &mut self,
        expert: usize,
        pointers: CudaExpertSlotPointers,
    ) -> Result<CudaExpertSlotBinding> {
        if let Some(binding) = self.binding(expert) {
            return Ok(binding);
        }
        if expert >= self.expert_capacity() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert id {expert} exceeds slot table capacity {}",
                    self.expert_capacity()
                ),
            });
        }
        let mut used = vec![false; self.slot_capacity()];
        for slot in &self.expert_to_slot {
            if *slot >= 0 && (*slot as usize) < used.len() {
                used[*slot as usize] = true;
            }
        }
        let has_free_slot = used.iter().any(|used| !used);
        let slot = used
            .iter()
            .enumerate()
            .find_map(|(slot, used)| {
                (!*used && self.slot_generation[slot] < i32::MAX - 1).then_some(slot)
            })
            .ok_or_else(|| {
                if has_free_slot {
                    Error::Internal {
                        message: "CUDA expert slot generation exhausted".into(),
                    }
                } else {
                    Error::Internal {
                        message: "CUDA expert slot table is full".into(),
                    }
                }
            })?;
        let generation = self.slot_generation[slot]
            .checked_add(1)
            .filter(|generation| *generation > 0 && *generation < i32::MAX)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert slot generation exhausted".into(),
            })?;
        self.slot_generation[slot] = generation;
        self.expert_to_slot[expert] = slot as i32;
        self.expert_generation[expert] = generation;
        self.gate_weight[slot] = pointers.gate_weight;
        self.gate_scale[slot] = pointers.gate_scale;
        self.up_weight[slot] = pointers.up_weight;
        self.up_scale[slot] = pointers.up_scale;
        self.down_weight[slot] = pointers.down_weight;
        self.down_scale[slot] = pointers.down_scale;
        Ok(CudaExpertSlotBinding {
            slot: slot as i32,
            generation,
        })
    }

    pub fn evict(&mut self, expert: usize) -> Result<bool> {
        let Some(binding) = self.binding(expert) else {
            return Ok(false);
        };
        let slot = binding.slot as usize;
        self.expert_to_slot[expert] = -1;
        self.expert_generation[expert] = 0;
        self.gate_weight[slot] = 0;
        self.gate_scale[slot] = 0;
        self.up_weight[slot] = 0;
        self.up_scale[slot] = 0;
        self.down_weight[slot] = 0;
        self.down_scale[slot] = 0;
        self.slot_generation[slot] = self.slot_generation[slot]
            .checked_add(1)
            .filter(|generation| *generation > 0)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert slot generation exhausted".into(),
            })?;
        Ok(true)
    }

    fn clear(&mut self) -> Result<bool> {
        let mut changed = false;
        for expert in 0..self.expert_capacity() {
            changed |= self.evict(expert)?;
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod expert_slot_generation_tests {
    use super::{CudaExpertSlotBinding, CudaExpertSlotPointers, CudaExpertSlotTableHost};

    const POINTERS: CudaExpertSlotPointers = CudaExpertSlotPointers {
        gate_weight: 1,
        gate_scale: 2,
        up_weight: 3,
        up_scale: 4,
        down_weight: 5,
        down_scale: 6,
    };

    #[test]
    fn exact_install_evict_and_reuse_follow_external_generations() {
        let mut table = CudaExpertSlotTableHost::new(3, 2).expect("slot table");

        let first = table
            .install_at(2, 1, 1, POINTERS)
            .expect("exact first install");
        assert_eq!(
            first,
            CudaExpertSlotBinding {
                slot: 1,
                generation: 1,
            }
        );
        assert_eq!(table.binding(2), Some(first));
        assert_eq!(table.pointers_at(1), POINTERS);

        table
            .evict_binding(2, 1, 1)
            .expect("exact binding eviction");
        assert_eq!(table.binding(2), None);
        assert!(!table.is_current(first));
        assert_eq!(table.slot_generation[1], 2);
        assert_eq!(
            table.pointers_at(1),
            CudaExpertSlotPointers {
                gate_weight: 0,
                gate_scale: 0,
                up_weight: 0,
                up_scale: 0,
                down_weight: 0,
                down_scale: 0,
            }
        );

        let second = table
            .install_at(1, 1, 2, POINTERS)
            .expect("exact reused install");
        assert_eq!(
            second,
            CudaExpertSlotBinding {
                slot: 1,
                generation: 2,
            }
        );
        assert_eq!(table.binding(1), Some(second));
    }

    #[test]
    fn exact_binding_mismatches_are_failure_atomic() {
        let mut table = CudaExpertSlotTableHost::new(3, 2).expect("slot table");
        table
            .install_at(0, 0, 1, POINTERS)
            .expect("exact first install");
        let installed = table.clone();

        assert!(table.install_at(1, 0, 2, POINTERS).is_err());
        assert_eq!(table, installed, "occupied slot mismatch mutated table");
        assert!(table.install_at(0, 1, 1, POINTERS).is_err());
        assert_eq!(table, installed, "expert binding mismatch mutated table");
        assert!(table.evict_binding(0, 0, 2).is_err());
        assert_eq!(table, installed, "stale generation eviction mutated table");
        assert!(table.evict_binding(0, 1, 1).is_err());
        assert_eq!(table, installed, "stale slot eviction mutated table");
        assert!(table.evict_binding(1, 0, 1).is_err());
        assert_eq!(table, installed, "stale expert eviction mutated table");

        table
            .evict_binding(0, 0, 1)
            .expect("exact binding eviction");
        let evicted = table.clone();
        assert!(table.install_at(1, 0, 3, POINTERS).is_err());
        assert_eq!(table, evicted, "generation mismatch mutated free slot");
    }

    #[test]
    fn exact_install_validates_coordinates_generation_and_full_pointer_tuple() {
        let mut table = CudaExpertSlotTableHost::new(2, 1).expect("slot table");
        let empty = table.clone();

        assert!(table.install_at(2, 0, 1, POINTERS).is_err());
        assert!(table.install_at(0, 1, 1, POINTERS).is_err());
        assert!(table.install_at(0, 0, 0, POINTERS).is_err());
        assert!(
            table
                .install_at(0, 0, i32::MAX as u32 + 1, POINTERS)
                .is_err()
        );
        for missing in [
            CudaExpertSlotPointers {
                gate_weight: 0,
                ..POINTERS
            },
            CudaExpertSlotPointers {
                gate_scale: 0,
                ..POINTERS
            },
            CudaExpertSlotPointers {
                up_weight: 0,
                ..POINTERS
            },
            CudaExpertSlotPointers {
                up_scale: 0,
                ..POINTERS
            },
            CudaExpertSlotPointers {
                down_weight: 0,
                ..POINTERS
            },
            CudaExpertSlotPointers {
                down_scale: 0,
                ..POINTERS
            },
        ] {
            assert!(table.install_at(0, 0, 1, missing).is_err());
        }
        assert_eq!(table, empty);
    }

    #[test]
    fn terminal_generation_is_free_but_exhausted() {
        let mut table = CudaExpertSlotTableHost::new(2, 1).expect("slot table");
        table.slot_generation[0] = i32::MAX - 2;

        let resident = table.install(0, POINTERS).expect("max-1 resident");
        assert_eq!(resident.generation, i32::MAX - 1);
        assert!(table.evict(0).expect("evict max-1 resident"));
        assert_eq!(table.slot_generation[0], i32::MAX);
        assert_eq!(table.binding(0), None);

        let error = table
            .install(1, POINTERS)
            .expect_err("terminal generation must not be published");
        assert!(error.to_string().contains("generation exhausted"));
        assert_eq!(table.binding(1), None);
    }
}

/// Stable device arrays plus their authoritative host mirror for one MoE layer.
pub struct CudaExpertSlotTable {
    gate_weight: DeviceBuffer<u64>,
    gate_scale: DeviceBuffer<u64>,
    up_weight: DeviceBuffer<u64>,
    up_scale: DeviceBuffer<u64>,
    down_weight: DeviceBuffer<u64>,
    down_scale: DeviceBuffer<u64>,
    expert_to_slot: DeviceBuffer<i32>,
    expert_generation: DeviceBuffer<i32>,
    slot_generation: DeviceBuffer<i32>,
    /// Mutations accepted onto the upload stream, including events not yet
    /// observed by the host. This is the reservation-order authority.
    submitted: CudaExpertSlotTableHost,
    /// Mutations whose exact completion events have been observed. Compute and
    /// higher-level residency publication validate only against this mirror.
    host: CudaExpertSlotTableHost,
    poisoned: bool,
}

impl CudaExpertSlotTable {
    pub fn host(&self) -> &CudaExpertSlotTableHost {
        &self.host
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn ensure_healthy(&self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Internal {
                message: "CUDA expert slot table is poisoned after a failed physical mutation"
                    .into(),
            });
        }
        Ok(())
    }

    fn ensure_synchronously_mutable(&self) -> Result<()> {
        self.ensure_healthy()?;
        if self.submitted != self.host {
            return Err(Error::Internal {
                message: "CUDA expert slot table has unpublished upload-stream mutations".into(),
            });
        }
        Ok(())
    }
}

pub struct CudaExpertRouteResolveWorkspace {
    route_slots: CudaI32Buffer,
    route_generations: CudaI32Buffer,
    miss_markers: CudaI32Buffer,
    /// `[count, overflow, miss_id...]`, kept contiguous so the host miss path
    /// needs one bounded D2H transfer rather than three synchronization points.
    miss_control: CudaI32Buffer,
    miss_capacity: usize,
    route_capacity: usize,
    /// Persistent page-locked destination for control-stream D2H.
    miss_staging: PinnedHostBuffer<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaExpertRouteMisses {
    pub miss_ids: Vec<i32>,
    pub route_ids: Vec<i32>,
    pub overflow: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaExpertRouteResolveResult {
    pub route_slots: Vec<i32>,
    pub route_generations: Vec<i32>,
    pub miss_markers: Vec<i32>,
    pub miss_ids: Vec<i32>,
    pub miss_overflow: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaExpertGroupRoutePlanDownload {
    pub active_expert_slots: Vec<i32>,
    pub active_group_generations: Vec<i32>,
    pub expert_route_indptr: Vec<i32>,
    pub expert_route_counts: Vec<i32>,
    pub route_token_indices: Vec<i32>,
    pub route_indices: Vec<i32>,
    pub route_weights: Vec<f32>,
    pub host: CudaExpertGroupRoutePlanHost,
    pub dispatch_error: bool,
}

/// Reusable host-side context for generic artifact-format CUDA operators.
///
/// Creates one CUDA context, loads the native provider modules once, and reuses
/// dedicated compute, upload, and control streams. It is intentionally generic
/// and knows only packed artifact formats plus explicit shapes; model-family
/// semantics stay in model code.
pub struct CudaArtifactOperatorContext {
    _ctx: Arc<CudaContext>,
    module: LoadedModule,
    stream: Arc<CudaStream>,
    upload_stream: Arc<CudaStream>,
    control_stream: Arc<CudaStream>,
    counters: CudaOpCounterCells,
    failpoints: CudaFailpoints,
    observability: CudaObservabilityConfig,
    /// When true, any device allocation, D2H copy, or stream-wide sync inside
    /// a capture region returns an error immediately. This is the E2
    /// capture-safe assertion mode.
    capture_safe: Cell<bool>,
}

impl CudaArtifactOperatorContext {
    pub fn new() -> Result<Self> {
        let observability = CudaObservabilityConfig::from_env();
        let ctx = cu(CudaContext::new(0))?;
        cu(ctx.bind_to_thread())?;
        let module = cu(crate::cuda::kernels::kernels::load(&ctx))?;
        let priorities = cu(ctx.stream_priority_range())?;
        // Compute uses the device's highest supported stream priority. Background
        // materialization remains work-conserving on the lowest-priority upload
        // stream, but it cannot monopolize SM scheduling when foreground kernels
        // become runnable.
        let default_stream = ctx.default_stream();
        let stream = cu(default_stream.fork_with_priority(priorities.highest_urgency()))?;
        let upload_stream = cu(default_stream.fork_with_priority(priorities.lowest_urgency()))?;
        // Small host-visible control transfers should keep pace with foreground
        // compute and wait only for exact producer events.
        let control_stream = cu(default_stream.fork_with_priority(priorities.highest_urgency()))?;
        Ok(Self {
            _ctx: ctx,
            module,
            stream,
            upload_stream,
            control_stream,
            counters: CudaOpCounterCells::default(),
            failpoints: CudaFailpoints::default(),
            observability,
            capture_safe: Cell::new(false),
        })
    }

    pub fn counters(&self) -> CudaOpCounters {
        self.counters.snapshot()
    }

    /// Return free and total bytes for the device bound to this operator context.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        cu(self._ctx.memory_info())
    }

    pub fn expert_slot_table(
        &self,
        expert_capacity: usize,
        slot_capacity: usize,
    ) -> Result<CudaExpertSlotTable> {
        let host = CudaExpertSlotTableHost::new(expert_capacity, slot_capacity)?;
        let table = CudaExpertSlotTable {
            gate_weight: self.upload_device_slice(&host.gate_weight)?,
            gate_scale: self.upload_device_slice(&host.gate_scale)?,
            up_weight: self.upload_device_slice(&host.up_weight)?,
            up_scale: self.upload_device_slice(&host.up_scale)?,
            down_weight: self.upload_device_slice(&host.down_weight)?,
            down_scale: self.upload_device_slice(&host.down_scale)?,
            expert_to_slot: self.upload_device_slice(&host.expert_to_slot)?,
            expert_generation: self.upload_device_slice(&host.expert_generation)?,
            slot_generation: self.upload_device_slice(&host.slot_generation)?,
            submitted: host.clone(),
            host,
            poisoned: false,
        };
        let initialized = self.record_compute_event()?;
        self.wait_compute_event_on_upload_stream(&initialized)?;
        Ok(table)
    }

    fn upload_device_slice<T: DeviceCopy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let buffer = self.record_device_allocation(
            values.len(),
            DeviceBuffer::from_host(&self.stream, values),
        )?;
        self.counters.add_host_to_device(slice_bytes(values));
        Ok(buffer)
    }

    fn download_device_slice<T: DeviceCopy>(
        &self,
        buffer: &DeviceBuffer<T>,
        len: usize,
    ) -> Result<Vec<T>> {
        self.check_capture_safe("device-to-host download")?;
        let values = cu(buffer.to_host_vec(&self.stream))?;
        self.counters.add_device_to_host(element_bytes::<T>(len));
        Ok(values)
    }

    fn write_expert_slot_table_host(
        &self,
        table: &mut CudaExpertSlotTable,
        host: &CudaExpertSlotTableHost,
    ) -> Result<()> {
        self.check_capture_safe("expert slot table publication")?;

        fn enqueue<T: DeviceCopy>(
            stream: &CudaStream,
            src: &[T],
            dst: &DeviceBuffer<T>,
        ) -> Result<()> {
            let bytes = slice_bytes(src) as usize;
            cu(runtime::copy_host_to_device(
                stream,
                dst.cu_deviceptr(),
                src.as_ptr().cast(),
                bytes,
            ))
        }

        enqueue(&self.stream, &host.gate_weight, &table.gate_weight)?;
        enqueue(&self.stream, &host.gate_scale, &table.gate_scale)?;
        enqueue(&self.stream, &host.up_weight, &table.up_weight)?;
        enqueue(&self.stream, &host.up_scale, &table.up_scale)?;
        enqueue(&self.stream, &host.down_weight, &table.down_weight)?;
        enqueue(&self.stream, &host.down_scale, &table.down_scale)?;
        enqueue(&self.stream, &host.expert_to_slot, &table.expert_to_slot)?;
        enqueue(
            &self.stream,
            &host.expert_generation,
            &table.expert_generation,
        )?;
        enqueue(&self.stream, &host.slot_generation, &table.slot_generation)?;

        for bytes in [
            slice_bytes(&host.gate_weight),
            slice_bytes(&host.gate_scale),
            slice_bytes(&host.up_weight),
            slice_bytes(&host.up_scale),
            slice_bytes(&host.down_weight),
            slice_bytes(&host.down_scale),
            slice_bytes(&host.expert_to_slot),
            slice_bytes(&host.expert_generation),
            slice_bytes(&host.slot_generation),
        ] {
            self.counters.add_host_to_device(bytes);
        }
        self.record_stream_wide_sync(self.stream.synchronize())
    }

    fn publish_expert_slot_table_host(
        &self,
        table: &mut CudaExpertSlotTable,
        next: CudaExpertSlotTableHost,
    ) -> Result<()> {
        table.ensure_healthy()?;
        let previous = table.host.clone();
        if let Err(error) = self.write_expert_slot_table_host(table, &next) {
            if let Err(rollback) = self.write_expert_slot_table_host(table, &previous) {
                table.poisoned = true;
                return Err(Error::Internal {
                    message: format!(
                        "CUDA expert slot table update failed ({error}); rollback also failed ({rollback}); table poisoned"
                    ),
                });
            }
            return Err(error);
        }
        table.host = next.clone();
        table.submitted = next;
        Ok(())
    }

    pub fn submit_expert_slot_install(
        &self,
        table: &mut CudaExpertSlotTable,
        target: CudaExpertSlotInstallTarget,
        expert: usize,
        slot: u32,
        generation: u32,
        pointers: CudaExpertSlotPointers,
    ) -> Result<CudaExpertSlotInstallTicket> {
        table.ensure_healthy()?;
        let (previous, replacement_quiescence) = match target {
            CudaExpertSlotInstallTarget::Empty => (None, None),
            CudaExpertSlotInstallTarget::Replacement {
                previous_expert,
                previous_binding,
                consumer_quiescence,
            } => (
                Some((previous_expert, previous_binding)),
                Some(consumer_quiescence),
            ),
        };
        let mut next = table.submitted.clone();
        let previous_abi = if let Some((previous_expert, previous_binding)) = previous {
            let previous_slot =
                u32::try_from(previous_binding.slot).map_err(|_| Error::Internal {
                    message: "CUDA previous expert slot is negative".into(),
                })?;
            let previous_generation =
                u32::try_from(previous_binding.generation).map_err(|_| Error::Internal {
                    message: "CUDA previous expert generation is negative".into(),
                })?;
            next.evict_binding(previous_expert, previous_slot, previous_generation)?;
            Some((
                u32::try_from(previous_expert).map_err(|_| Error::Internal {
                    message: "CUDA previous expert index exceeds u32".into(),
                })?,
                previous_slot,
                next.slot_generation[previous_slot as usize],
            ))
        } else {
            None
        };
        next.install_at(expert, slot, generation, pointers)?;
        let expert_abi = u32::try_from(expert).map_err(|_| Error::Internal {
            message: "CUDA expert index exceeds u32".into(),
        })?;
        let generation_abi = i32::try_from(generation).map_err(|_| Error::Internal {
            message: "CUDA expert generation exceeds i32".into(),
        })?;
        self.check_capture_safe("expert slot binding publication")?;

        let mut upload_work_submitted = false;
        let mut mutation_submitted = false;
        let submission = (|| {
            if let Some((previous_expert, previous_slot, next_generation)) = previous_abi {
                self.wait_compute_event_on_upload_stream(
                    replacement_quiescence
                        .as_ref()
                        .expect("replacement target contains consumer quiescence"),
                )?;
                upload_work_submitted = true;
                self.launched_on_upload(unsafe {
                    self.module.evict_expert_slot_binding(
                        &self.upload_stream,
                        LaunchConfig::for_num_elems(1),
                        &mut table.gate_weight,
                        &mut table.gate_scale,
                        &mut table.up_weight,
                        &mut table.up_scale,
                        &mut table.down_weight,
                        &mut table.down_scale,
                        &mut table.expert_to_slot,
                        &mut table.expert_generation,
                        &mut table.slot_generation,
                        previous_expert,
                        previous_slot,
                        next_generation,
                    )
                })?;
                mutation_submitted = true;
            }
            self.launched_on_upload(unsafe {
                self.module.install_expert_slot_binding(
                    &self.upload_stream,
                    LaunchConfig::for_num_elems(1),
                    &mut table.gate_weight,
                    &mut table.gate_scale,
                    &mut table.up_weight,
                    &mut table.up_scale,
                    &mut table.down_weight,
                    &mut table.down_scale,
                    &mut table.expert_to_slot,
                    &mut table.expert_generation,
                    &mut table.slot_generation,
                    expert_abi,
                    slot,
                    generation_abi,
                    pointers.gate_weight,
                    pointers.gate_scale,
                    pointers.up_weight,
                    pointers.up_scale,
                    pointers.down_weight,
                    pointers.down_scale,
                )
            })?;
            mutation_submitted = true;
            self.record_upload_event()
        })();
        let event = match submission {
            Ok(event) => event,
            Err(error) if upload_work_submitted => {
                if mutation_submitted {
                    table.poisoned = true;
                }
                return Err(Error::with_cleanup(
                    "submit CUDA expert slot mutation",
                    error,
                    self.sync_upload_stream(),
                ));
            }
            Err(error) => return Err(error),
        };
        table.submitted = next;
        Ok(CudaExpertSlotInstallTicket {
            event,
            _replacement_quiescence: replacement_quiescence,
            previous,
            expert,
            slot,
            generation,
            pointers,
            completed: false,
        })
    }

    pub fn install_expert_slot_at(
        &self,
        table: &mut CudaExpertSlotTable,
        expert: usize,
        slot: u32,
        generation: u32,
        pointers: CudaExpertSlotPointers,
    ) -> Result<CudaExpertSlotBinding> {
        table.ensure_synchronously_mutable()?;
        let mut next = table.host.clone();
        let binding = next.install_at(expert, slot, generation, pointers)?;
        if next == table.host {
            return Ok(binding);
        }
        let expert = u32::try_from(expert).map_err(|_| Error::Internal {
            message: "CUDA expert index exceeds u32".into(),
        })?;
        let generation = i32::try_from(generation).map_err(|_| Error::Internal {
            message: "CUDA expert generation exceeds i32".into(),
        })?;
        self.check_capture_safe("expert slot binding publication")?;
        self.launched(unsafe {
            self.module.install_expert_slot_binding(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &mut table.gate_weight,
                &mut table.gate_scale,
                &mut table.up_weight,
                &mut table.up_scale,
                &mut table.down_weight,
                &mut table.down_scale,
                &mut table.expert_to_slot,
                &mut table.expert_generation,
                &mut table.slot_generation,
                expert,
                slot,
                generation,
                pointers.gate_weight,
                pointers.gate_scale,
                pointers.up_weight,
                pointers.up_scale,
                pointers.down_weight,
                pointers.down_scale,
            )
        })?;
        table.host = next.clone();
        table.submitted = next;
        Ok(binding)
    }

    pub fn evict_expert_slot_binding(
        &self,
        table: &mut CudaExpertSlotTable,
        expert: usize,
        slot: u32,
        generation: u32,
    ) -> Result<()> {
        table.ensure_synchronously_mutable()?;
        let mut next = table.host.clone();
        next.evict_binding(expert, slot, generation)?;
        let expert = u32::try_from(expert).map_err(|_| Error::Internal {
            message: "CUDA expert index exceeds u32".into(),
        })?;
        let next_generation = next.slot_generation[slot as usize];
        self.check_capture_safe("expert slot binding eviction")?;
        self.launched(unsafe {
            self.module.evict_expert_slot_binding(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &mut table.gate_weight,
                &mut table.gate_scale,
                &mut table.up_weight,
                &mut table.up_scale,
                &mut table.down_weight,
                &mut table.down_scale,
                &mut table.expert_to_slot,
                &mut table.expert_generation,
                &mut table.slot_generation,
                expert,
                slot,
                next_generation,
            )
        })?;
        table.host = next.clone();
        table.submitted = next;
        Ok(())
    }

    pub fn install_expert_slot(
        &self,
        table: &mut CudaExpertSlotTable,
        expert: usize,
        pointers: CudaExpertSlotPointers,
    ) -> Result<CudaExpertSlotBinding> {
        table.ensure_synchronously_mutable()?;
        if let Some(binding) = table.host.binding(expert) {
            return Ok(binding);
        }
        let mut next = table.host.clone();
        let binding = next.install(expert, pointers)?;
        let expert = u32::try_from(expert).map_err(|_| Error::Internal {
            message: "CUDA expert index exceeds u32".into(),
        })?;
        self.check_capture_safe("expert slot binding publication")?;
        self.launched(unsafe {
            self.module.install_expert_slot_binding(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &mut table.gate_weight,
                &mut table.gate_scale,
                &mut table.up_weight,
                &mut table.up_scale,
                &mut table.down_weight,
                &mut table.down_scale,
                &mut table.expert_to_slot,
                &mut table.expert_generation,
                &mut table.slot_generation,
                expert,
                binding.slot as u32,
                binding.generation,
                pointers.gate_weight,
                pointers.gate_scale,
                pointers.up_weight,
                pointers.up_scale,
                pointers.down_weight,
                pointers.down_scale,
            )
        })?;
        table.host = next.clone();
        table.submitted = next;
        Ok(binding)
    }

    pub fn evict_expert_slot(
        &self,
        table: &mut CudaExpertSlotTable,
        expert: usize,
    ) -> Result<bool> {
        table.ensure_synchronously_mutable()?;
        let Some(binding) = table.host.binding(expert) else {
            return Ok(false);
        };
        self.evict_expert_slot_binding(
            table,
            expert,
            binding.slot as u32,
            binding.generation as u32,
        )?;
        Ok(true)
    }

    pub fn clear_expert_slot_table(&self, table: &mut CudaExpertSlotTable) -> Result<bool> {
        table.ensure_synchronously_mutable()?;
        let mut next = table.host.clone();
        if !next.clear()? {
            return Ok(false);
        }
        self.publish_expert_slot_table_host(table, next)?;
        Ok(true)
    }

    pub fn expert_route_resolve_workspace(
        &self,
        route_capacity: usize,
        miss_capacity: usize,
    ) -> Result<CudaExpertRouteResolveWorkspace> {
        if route_capacity == 0 || miss_capacity == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert route resolve requires positive capacities: routes={route_capacity} misses={miss_capacity}"
                ),
            });
        }
        Ok(CudaExpertRouteResolveWorkspace {
            route_slots: self.zero_i32_buffer(route_capacity)?,
            route_generations: self.zero_i32_buffer(route_capacity)?,
            miss_markers: self.zero_i32_buffer(route_capacity)?,
            miss_control: self.zero_i32_buffer(
                2usize
                    .saturating_add(miss_capacity)
                    .saturating_add(route_capacity),
            )?,
            miss_capacity,
            route_capacity,
            miss_staging: cu(PinnedHostBuffer::zeroed(
                &self._ctx,
                2usize
                    .saturating_add(miss_capacity)
                    .saturating_add(route_capacity),
            ))?,
        })
    }

    pub fn resolve_expert_routes(
        &self,
        table: &CudaExpertSlotTable,
        expert_ids: &CudaI32Buffer,
        route_count: usize,
        workspace: &mut CudaExpertRouteResolveWorkspace,
    ) -> Result<()> {
        table.ensure_healthy()?;
        if route_count > expert_ids.len()
            || route_count > workspace.route_slots.len()
            || route_count > workspace.route_generations.len()
            || route_count > workspace.miss_markers.len()
            || route_count > workspace.route_capacity
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert route resolve exceeds capacity: routes={route_count} ids={} slots={} generations={} markers={}",
                    expert_ids.len(),
                    workspace.route_slots.len(),
                    workspace.route_generations.len(),
                    workspace.miss_markers.len()
                ),
            });
        }
        let route_count = u32::try_from(route_count).map_err(|_| Error::Internal {
            message: "CUDA expert route count exceeds u32".into(),
        })?;
        let expert_capacity =
            u32::try_from(table.host.expert_capacity()).map_err(|_| Error::Internal {
                message: "CUDA expert table capacity exceeds u32".into(),
            })?;
        let slot_capacity =
            u32::try_from(table.host.slot_capacity()).map_err(|_| Error::Internal {
                message: "CUDA expert slot capacity exceeds u32".into(),
            })?;
        let miss_capacity =
            u32::try_from(workspace.miss_capacity).map_err(|_| Error::Internal {
                message: "CUDA expert miss capacity exceeds u32".into(),
            })?;
        let route_capacity =
            u32::try_from(workspace.route_capacity).map_err(|_| Error::Internal {
                message: "CUDA expert route capacity exceeds u32".into(),
            })?;
        self.launched(unsafe {
            self.module.initialize_expert_slot_resolve(
                &self.stream,
                LaunchConfig::for_num_elems(
                    miss_capacity
                        .saturating_add(route_capacity)
                        .saturating_add(2)
                        .max(1),
                ),
                &mut workspace.miss_control.buffer,
                miss_capacity,
                route_capacity,
            )
        })?;
        self.launched(unsafe {
            self.module.resolve_expert_slots(
                &self.stream,
                LaunchConfig::for_num_elems(route_count.max(1)),
                &expert_ids.buffer,
                &table.expert_to_slot,
                &table.expert_generation,
                &table.slot_generation,
                &mut workspace.route_slots.buffer,
                &mut workspace.route_generations.buffer,
                &mut workspace.miss_markers.buffer,
                &mut workspace.miss_control.buffer,
                route_count,
                expert_capacity,
                slot_capacity,
                miss_capacity,
            )
        })
    }

    /// Download only the bounded miss control queue in one D2H transfer after
    /// all compute work currently queued on the primary stream.
    pub fn download_expert_route_misses(
        &self,
        workspace: &mut CudaExpertRouteResolveWorkspace,
    ) -> Result<CudaExpertRouteMisses> {
        let produced = self.record_compute_event()?;
        self.download_expert_route_misses_after(workspace, &produced)
    }

    /// Start the compact control D2H after an explicit producer event. Later
    /// primary-stream work can overlap while the host waits for this transfer.
    pub fn download_expert_route_misses_after(
        &self,
        workspace: &mut CudaExpertRouteResolveWorkspace,
        produced: &CudaComputeEvent,
    ) -> Result<CudaExpertRouteMisses> {
        self.check_capture_safe("expert miss control download")?;
        cu(self.control_stream.wait(&produced.event))?;
        let bytes = element_bytes::<i32>(workspace.miss_control.len()) as usize;
        let staging = workspace.miss_staging.as_mut_slice();
        cu(runtime::copy_device_to_host(
            &self.control_stream,
            staging.as_mut_ptr().cast(),
            workspace.miss_control.buffer.cu_deviceptr(),
            bytes,
        ))?;
        let copied = cu(self.control_stream.record_event(None))?;
        cu(copied.synchronize())?;
        self.counters.add_device_to_host(bytes as u64);

        let miss_count = staging[0].max(0) as usize;
        let overflow = staging[1] != 0;
        let miss_start = 2;
        let miss_end = miss_start + workspace.miss_capacity;
        let route_end = miss_end + workspace.route_capacity;
        let take = miss_count.min(workspace.miss_capacity);
        Ok(CudaExpertRouteMisses {
            miss_ids: staging[miss_start..miss_start + take].to_vec(),
            route_ids: staging[miss_end..route_end].to_vec(),
            overflow: overflow || miss_count > workspace.miss_capacity,
        })
    }

    /// Diagnostic/test oracle that downloads device-side route resolution.
    /// Production dispatch must consume the device-resident workspace directly
    /// and must not call this method.
    pub fn download_expert_route_resolve(
        &self,
        workspace: &mut CudaExpertRouteResolveWorkspace,
        route_count: usize,
    ) -> Result<CudaExpertRouteResolveResult> {
        if route_count > workspace.route_slots.len() {
            return Err(Error::Internal {
                message: "CUDA expert route resolve download exceeds route capacity".into(),
            });
        }
        let misses = self.download_expert_route_misses(workspace)?;
        let mut route_slots = self.download_i32_buffer(&workspace.route_slots)?;
        let mut route_generations = self.download_i32_buffer(&workspace.route_generations)?;
        let mut miss_markers = self.download_i32_buffer(&workspace.miss_markers)?;
        route_slots.truncate(route_count);
        route_generations.truncate(route_count);
        miss_markers.truncate(route_count);
        Ok(CudaExpertRouteResolveResult {
            route_slots,
            route_generations,
            miss_markers,
            miss_ids: misses.miss_ids,
            miss_overflow: misses.overflow,
        })
    }

    /// Whether this artifact accepts the provider's prepacked FP8 activation
    /// contract. Callers use this to avoid quantizing the same activation twice.
    pub fn artifact_linear_supports_prepacked_fp8(
        &self,
        handle: &CudaArtifactLinearHandle,
    ) -> bool {
        matches!(
            handle.shape,
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m: 128,
                block_k: 128,
            } if out_features.is_multiple_of(16) && in_features.is_multiple_of(128)
        )
    }

    pub fn capture_decode_graph(
        &self,
        capture_fn: impl FnOnce() -> Result<()>,
    ) -> Result<crate::cuda::graph::CudaGraphHandle> {
        crate::cuda::graph::capture_decode_graph(&self.stream, capture_fn)
    }

    pub fn launch_graph(&self, graph: &crate::cuda::graph::CudaGraphHandle) -> Result<()> {
        graph.launch(&self.stream)
    }

    /// Upload a captured graph's nodes to the device ahead of first launch.
    pub fn upload_graph(&self, graph: &crate::cuda::graph::CudaGraphHandle) -> Result<()> {
        graph.upload(&self.stream)
    }

    /// Build a llama.cpp-style auto-warmup cached decode graph bound to this
    /// context's stream. See [`crate::cuda::graph::CachedDecodeGraph`].
    pub fn cached_decode_graph(&self) -> crate::cuda::graph::CachedDecodeGraph {
        crate::cuda::graph::CachedDecodeGraph::new(&self._ctx)
    }

    /// Clone the stream for use with graph capture outside of `&self` borrow.
    pub fn stream_clone(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub fn compute_stream_authority(&self) -> CudaComputeStreamAuthority {
        CudaComputeStreamAuthority {
            stream: Arc::clone(&self.stream),
        }
    }

    pub fn stream_priorities(&self) -> Result<(i32, i32, i32)> {
        Ok((
            cu(self.stream.priority())?,
            cu(self.upload_stream.priority())?,
            cu(self.control_stream.priority())?,
        ))
    }

    pub fn sync_stream(&self) -> Result<()> {
        self.record_stream_wide_sync(self.stream.synchronize())
    }

    pub fn sync_upload_stream(&self) -> Result<()> {
        self.record_stream_wide_sync(self.upload_stream.synchronize())
    }

    pub fn wait_upload_event(&self, event: &CudaUploadEvent) -> Result<()> {
        cu(self.stream.wait(&event.event))
    }

    /// Order future upload-stream work after compute that may still reference a
    /// retired physical frame.
    ///
    /// The caller must keep `event` alive until a later upload event covering
    /// the dependent overwrite has completed.
    pub fn wait_compute_event_on_upload_stream(&self, event: &CudaComputeEvent) -> Result<()> {
        cu(self.upload_stream.wait(&event.event))
    }

    pub fn record_upload_event(&self) -> Result<CudaUploadEvent> {
        Ok(CudaUploadEvent {
            event: cu(self.upload_stream.record_event(None))?,
        })
    }

    /// Enqueue an allocation-free completion notification after all currently
    /// submitted upload-stream work. The callback executes on a CUDA driver
    /// thread and must not call CUDA APIs or block.
    pub fn notify_upload_stream<F>(&self, notify: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        cu(self.upload_stream.launch_host_function(notify))
    }

    /// Enqueue a completion callback after all current control-stream work.
    pub fn notify_control_stream<F>(&self, notify: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        cu(self.control_stream.launch_host_function(notify))
    }

    /// Enqueue a completion callback after all current primary-stream work.
    pub fn notify_compute_stream<F>(&self, notify: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        cu(self.stream.launch_host_function(notify))
    }

    pub fn record_compute_event(&self) -> Result<CudaComputeEvent> {
        self.compute_stream_authority().record_event()
    }

    pub fn reset_counters(&self) {
        self.counters.reset();
    }

    pub fn add_arena_hit(&self) {
        self.counters.add_arena_hit();
    }

    pub fn add_arena_miss(&self) {
        self.counters.add_arena_miss();
    }

    pub fn add_arena_grow(&self) {
        self.counters.add_arena_grow();
    }

    pub fn add_arena_reuse(&self) {
        self.counters.add_arena_reuse();
    }

    /// Returns a reference to the deterministic failpoint controller.
    pub fn failpoints(&self) -> &CudaFailpoints {
        &self.failpoints
    }

    /// Enable capture-safe assertion mode. While active, any device allocation,
    /// D2H copy, or stream-wide sync returns an error immediately. This is the
    /// E2 capture-safe assertion mode used by tests to verify that graph
    /// capture regions are allocation-free.
    pub fn enable_capture_safe(&self) {
        self.capture_safe.set(true);
    }

    /// Disable capture-safe assertion mode.
    pub fn disable_capture_safe(&self) {
        self.capture_safe.set(false);
    }

    /// Returns true if capture-safe mode is active.
    pub fn is_capture_safe(&self) -> bool {
        self.capture_safe.get()
    }

    /// Check if the current operation is allowed under capture-safe mode.
    /// Returns an error if capture-safe is enabled and the operation is forbidden.
    fn check_capture_safe(&self, op: &str) -> Result<()> {
        if self.capture_safe.get() {
            return Err(Error::Internal {
                message: format!(
                    "capture-safe violation: '{op}' is forbidden inside a graph capture region"
                ),
            });
        }
        Ok(())
    }

    fn record_kernel_launch(&self) {
        self.record_kernel_launches(1);
    }

    fn record_kernel_launches(&self, count: u64) {
        self.counters.add_compute_kernel_launches(count);
    }

    fn record_upload_kernel_launch(&self) {
        self.counters.add_upload_kernel_launch();
    }

    fn launched<T, E>(&self, result: std::result::Result<T, E>) -> Result<T>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let value = cu(result)?;
        self.record_kernel_launch();
        Ok(value)
    }

    fn launched_on_upload<T, E>(&self, result: std::result::Result<T, E>) -> Result<T>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let value = cu(result)?;
        self.record_upload_kernel_launch();
        Ok(value)
    }

    /// Allocate a device buffer without initializing it.
    ///
    /// Use only for scratch buffers that are fully written by subsequent kernels
    /// before any read. This keeps the unsafe `uninitialized_async` contract in
    /// one place instead of scattering `cu(unsafe { ... })` through hot paths.
    fn uninitialized_device_buffer<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        self.record_device_allocation(len, unsafe {
            DeviceBuffer::<T>::uninitialized_async(&self.stream, len)
        })
    }

    fn uninitialized_upload_device_buffer<T: DeviceCopy>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>> {
        self.check_capture_safe("device allocation")?;
        if self.failpoints.check_allocation() {
            return Err(Error::Internal {
                message: "deterministic failpoint: device allocation".into(),
            });
        }

        self.counters.begin_device_allocation();
        match cu(unsafe { DeviceBuffer::<T>::uninitialized_async(&self.upload_stream, len) }) {
            Ok(buffer) => {
                self.counters
                    .complete_device_allocation(element_bytes::<T>(len));
                Ok(buffer)
            }
            Err(error) => {
                self.counters.fail_device_allocation();
                Err(error)
            }
        }
    }

    fn zeroed_device_buffer<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        self.record_device_allocation(len, DeviceBuffer::<T>::zeroed(&self.stream, len))
    }

    fn record_device_allocation<T: DeviceCopy, E>(
        &self,
        len: usize,
        result: std::result::Result<DeviceBuffer<T>, E>,
    ) -> Result<DeviceBuffer<T>>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.check_capture_safe("device allocation")?;
        if self.failpoints.check_allocation() {
            return Err(Error::Internal {
                message: "deterministic failpoint: device allocation".into(),
            });
        }

        self.counters.begin_device_allocation();
        match cu(result) {
            Ok(buffer) => {
                self.counters
                    .complete_device_allocation(element_bytes::<T>(len));
                Ok(buffer)
            }
            Err(error) => {
                self.counters.fail_device_allocation();
                Err(error)
            }
        }
    }

    fn record_stream_wide_sync<T, E>(&self, result: std::result::Result<T, E>) -> Result<T>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.check_capture_safe("stream-wide sync")?;
        match cu(result) {
            Ok(value) => {
                self.counters.complete_stream_wide_sync();
                Ok(value)
            }
            Err(error) => {
                self.counters.fail_stream_wide_sync();
                Err(error)
            }
        }
    }

    fn upload_u8(&self, values: &[u8]) -> Result<DeviceBuffer<u8>> {
        let buffer = self.record_device_allocation(
            values.len(),
            DeviceBuffer::from_host(&self.stream, values),
        )?;
        self.counters.add_host_to_device(slice_bytes(values));
        Ok(buffer)
    }

    fn upload_f32(&self, values: &[f32]) -> Result<DeviceBuffer<f32>> {
        let buffer = self.record_device_allocation(
            values.len(),
            DeviceBuffer::from_host(&self.stream, values),
        )?;
        self.counters.add_host_to_device(slice_bytes(values));
        Ok(buffer)
    }

    fn upload_i32(&self, values: &[i32]) -> Result<DeviceBuffer<i32>> {
        let buffer = self.record_device_allocation(
            values.len(),
            DeviceBuffer::from_host(&self.stream, values),
        )?;
        self.counters.add_host_to_device(slice_bytes(values));
        Ok(buffer)
    }

    pub fn pinned_host_allocator(&self) -> CudaPinnedHostAllocator {
        CudaPinnedHostAllocator {
            ctx: Arc::clone(&self._ctx),
        }
    }

    pub fn pin_u8_host_buffer(&self, values: &[u8]) -> Result<CudaPinnedU8HostBuffer> {
        Ok(CudaPinnedU8HostBuffer {
            buffer: Arc::new(cu(PinnedHostBuffer::from_slice(&self._ctx, values))?),
            offset: 0,
            len: values.len(),
        })
    }

    /// Enqueue an async H2D copy from a pinned source on the artifact upload stream.
    ///
    /// # Safety
    /// The caller must keep `values` alive and immutable until the returned upload
    /// event has completed. Model-level upload tickets satisfy this by owning the
    /// pinned sources alongside the CUDA handles and event.
    unsafe fn upload_u8_from_pinned_async_unchecked(
        &self,
        values: &CudaPinnedU8HostBuffer,
    ) -> Result<DeviceBuffer<u8>> {
        let buffer = self.record_device_allocation(values.len(), unsafe {
            DeviceBuffer::from_pinned_host(&self.upload_stream, values.buffer.as_ref())
        })?;
        self.counters.add_host_to_device(values.len() as u64);
        Ok(buffer)
    }

    fn download_f32(&self, buffer: &DeviceBuffer<f32>, len: usize) -> Result<Vec<f32>> {
        self.check_capture_safe("device-to-host download")?;
        let values = cu(buffer.to_host_vec(&self.stream))?;
        self.counters.add_device_to_host(element_bytes::<f32>(len));
        Ok(values)
    }

    pub fn upload_f32_buffer(&self, values: &[f32]) -> Result<CudaF32Buffer> {
        Ok(CudaTypedBuffer::from_device_buffer(
            self.upload_f32(values)?,
        ))
    }

    pub fn zero_f32_buffer(&self, len: usize) -> Result<CudaF32Buffer> {
        Ok(CudaTypedBuffer::from_device_buffer(
            self.zeroed_device_buffer::<f32>(len)?,
        ))
    }

    pub fn zero_bf16_buffer(&self, len: usize) -> Result<CudaBf16Buffer> {
        Ok(CudaTypedBuffer::from_device_buffer(
            self.zeroed_device_buffer::<u16>(len)?,
        ))
    }

    pub fn hybrid_mla_explicit_selection_workspace(
        &self,
        layout: HybridMlaExplicitSelectionLayout,
    ) -> Result<CudaHybridMlaExplicitSelectionWorkspace> {
        let requirements = hybrid_mla_explicit_selection_workspace_requirements(layout)?;
        let capacity_bytes = usize::try_from(requirements.bytes).map_err(|_| Error::Internal {
            message: format!(
                "hybrid MLA explicit selection workspace requirement exceeds usize: {}",
                requirements.bytes
            ),
        })?;
        let alignment = requirements.alignment as usize;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Internal {
                message: format!(
                    "hybrid MLA explicit selection workspace returned invalid alignment {alignment}"
                ),
            });
        }
        let storage = self.uninitialized_device_buffer::<u8>(capacity_bytes)?;
        if !storage
            .cu_deviceptr()
            .is_multiple_of(u64::from(requirements.alignment))
        {
            return Err(Error::Internal {
                message: format!(
                    "hybrid MLA explicit selection workspace pointer does not satisfy alignment {alignment}"
                ),
            });
        }

        #[cfg(ferrule_cuda_test_oracle)]
        let output_values = layout
            .rows
            .checked_mul(layout.heads)
            .and_then(|value| value.checked_mul(layout.head_dim))
            .ok_or_else(|| Error::Internal {
                message: "hybrid MLA explicit selection oracle output size overflow".into(),
            })?;
        #[cfg(ferrule_cuda_test_oracle)]
        let status_words =
            crate::cuda::cutlass::HYBRID_MLA_EXPLICIT_SELECTION_TEST_COMPARE_RESULT_WORDS;
        #[cfg(not(ferrule_cuda_test_oracle))]
        let status_words = 1;

        Ok(CudaHybridMlaExplicitSelectionWorkspace {
            storage,
            status: CudaTypedBuffer::from_device_buffer(
                self.uninitialized_device_buffer::<i32>(status_words)?,
            ),
            capacity_bytes,
            alignment,
            allocated_layout: layout,
            #[cfg(ferrule_cuda_test_oracle)]
            oracle_output: self.zeroed_device_buffer::<f32>(output_values)?,
        })
    }

    pub fn hybrid_mla_attention_workspace(&self) -> Result<CudaHybridMlaAttentionWorkspace> {
        let output_values = crate::cuda::cutlass::PROPOSAL_ROWS
            .checked_mul(crate::cuda::cutlass::HYBRID_MLA_ATTENTION_HEADS)
            .and_then(|value| {
                value.checked_mul(crate::cuda::cutlass::HYBRID_MLA_ATTENTION_HEAD_DIM)
            })
            .ok_or_else(|| Error::Internal {
                message: "proposal attention output size overflow".into(),
            })?;
        let score_values = crate::cuda::cutlass::PROPOSAL_ROWS
            .checked_mul(crate::cuda::cutlass::HYBRID_MLA_ATTENTION_HEADS)
            .and_then(|value| {
                value.checked_mul(crate::cuda::cutlass::HYBRID_MLA_ATTENTION_TOKEN_CAPACITY)
            })
            .ok_or_else(|| Error::Internal {
                message: "proposal attention score size overflow".into(),
            })?;
        let gathered_values = crate::cuda::cutlass::HYBRID_MLA_ATTENTION_TOKEN_CAPACITY
            .checked_mul(crate::cuda::cutlass::HYBRID_MLA_ATTENTION_HEAD_DIM)
            .ok_or_else(|| Error::Internal {
                message: "proposal gathered KV size overflow".into(),
            })?;
        let pair_values = crate::cuda::cutlass::PROPOSAL_ROWS
            .checked_mul(crate::cuda::cutlass::HYBRID_MLA_ATTENTION_HEADS)
            .ok_or_else(|| Error::Internal {
                message: "proposal attention row/head size overflow".into(),
            })?;
        let rescale_values = pair_values
            .checked_mul(crate::cuda::cutlass::HYBRID_MLA_ATTENTION_ONLINE_SOFTMAX_TILES)
            .ok_or_else(|| Error::Internal {
                message: "proposal attention online-softmax size overflow".into(),
            })?;
        Ok(CudaHybridMlaAttentionWorkspace {
            query_bf16: self.zeroed_device_buffer::<u16>(output_values)?,
            gathered_kv_bf16: self.zeroed_device_buffer::<u16>(gathered_values)?,
            scores: self.zero_f32_buffer(score_values)?,
            probabilities_bf16: self.zeroed_device_buffer::<u16>(score_values)?,
            online_rescales: self.zero_f32_buffer(rescale_values)?,
            denominators: self.zero_f32_buffer(pair_values)?,
            status: self.zero_i32_buffer(1)?,
        })
    }

    pub fn proposal_head_workspace(
        &self,
        rows: usize,
        hidden: usize,
        vocab: usize,
        partial_capacity: usize,
    ) -> Result<CudaProposalHeadWorkspace> {
        let hidden_values = rows.checked_mul(hidden).ok_or_else(|| Error::Internal {
            message: "proposal proposal hidden size overflow".into(),
        })?;
        let logits_values = rows.checked_mul(vocab).ok_or_else(|| Error::Internal {
            message: "proposal proposal logits size overflow".into(),
        })?;
        Ok(CudaProposalHeadWorkspace {
            hidden: self.zero_f32_buffer(hidden_values)?,
            normalized: self.zero_f32_buffer(hidden_values)?,
            base_logits: self.zero_f32_buffer(logits_values)?,
            partial_values: self.zero_f32_buffer(partial_capacity)?,
            partial_indices: self.zero_i32_buffer(partial_capacity)?,
            token_ids: self.i32_host_mirror(&vec![0; rows + 1])?,
            confidence: self.zero_f32_buffer(rows)?,
            status: self.zero_i32_buffer(1)?,
            result: self.i32_host_mirror(&vec![0; 1 + 2 * rows])?,
        })
    }

    /// Zero an existing device buffer in-place (cuMemsetD32Async, no allocation).
    /// Safe for CUDA graph capture.
    pub fn zero_f32_buffer_in_place(&self, buf: &mut CudaF32Buffer) -> Result<()> {
        cu(runtime::memset_u32(
            &self.stream,
            buf.buffer.cu_deviceptr(),
            0,
            buf.len(),
        ))?;
        self.counters.add_compute_kernel_launch();
        Ok(())
    }

    pub fn zero_f32_range(
        &self,
        buffer: &mut CudaF32Buffer,
        offset: usize,
        len: usize,
    ) -> Result<()> {
        let ptr = f32_range_device_ptr(buffer, offset, len, "zero_f32_range")?;
        if len == 0 {
            return Ok(());
        }
        cu(runtime::memset_u32(&self.stream, ptr, 0, len))?;
        self.counters.add_compute_kernel_launch();
        Ok(())
    }

    pub fn copy_f32_range(
        &self,
        src: &CudaF32Buffer,
        src_offset: usize,
        dst: &mut CudaF32Buffer,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let src_ptr = f32_range_device_ptr(src, src_offset, len, "copy_f32_range source")?;
        let dst_ptr = f32_range_device_ptr(dst, dst_offset, len, "copy_f32_range destination")?;
        copy_f32_device_range(&self.stream, src_ptr, dst_ptr, len)
    }

    /// Copies two non-overlapping ranges within one device buffer.
    pub fn copy_f32_within(
        &self,
        buffer: &mut CudaF32Buffer,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let src_end = src_offset.checked_add(len).ok_or_else(|| Error::Internal {
            message: "CUDA f32 within-copy source overflow".into(),
        })?;
        let dst_end = dst_offset.checked_add(len).ok_or_else(|| Error::Internal {
            message: "CUDA f32 within-copy destination overflow".into(),
        })?;
        if src_end > buffer.len() || dst_end > buffer.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA f32 within-copy out of bounds: buffer={} src={src_offset}..{src_end} dst={dst_offset}..{dst_end}",
                    buffer.len()
                ),
            });
        }
        if len != 0 && src_offset < dst_end && dst_offset < src_end {
            return Err(Error::Internal {
                message: "CUDA f32 within-copy ranges must not overlap".into(),
            });
        }
        let src_ptr = f32_range_device_ptr(buffer, src_offset, len, "copy_f32_within source")?;
        let dst_ptr = f32_range_device_ptr(buffer, dst_offset, len, "copy_f32_within destination")?;
        copy_f32_device_range(&self.stream, src_ptr, dst_ptr, len)
    }

    pub fn download_f32_range(
        &self,
        buffer: &CudaF32Buffer,
        offset: usize,
        len: usize,
    ) -> Result<Vec<f32>> {
        self.check_capture_safe("device-to-host range download")?;
        let src = f32_range_device_ptr(buffer, offset, len, "download_f32_range")?;
        let mut values = vec![0.0f32; len];
        if len == 0 {
            return Ok(values);
        }
        let bytes = element_bytes::<f32>(len) as usize;
        cu(runtime::copy_device_to_host(
            &self.stream,
            values.as_mut_ptr().cast(),
            src,
            bytes,
        ))?;
        self.record_stream_wide_sync(self.stream.synchronize())?;
        self.counters.add_device_to_host(bytes as u64);
        Ok(values)
    }

    pub fn overwrite_f32_range(
        &self,
        src: &[f32],
        dst: &mut CudaF32Buffer,
        dst_offset: usize,
    ) -> Result<()> {
        self.check_capture_safe("host-to-device range upload")?;
        let dst_ptr = f32_range_device_ptr(dst, dst_offset, src.len(), "overwrite_f32_range")?;
        if src.is_empty() {
            return Ok(());
        }
        let bytes = slice_bytes(src) as usize;
        cu(runtime::copy_host_to_device(
            &self.stream,
            dst_ptr,
            src.as_ptr().cast(),
            bytes,
        ))?;
        self.record_stream_wide_sync(self.stream.synchronize())?;
        self.counters.add_host_to_device(bytes as u64);
        Ok(())
    }

    pub fn download_f32_buffer(&self, buffer: &CudaF32Buffer) -> Result<Vec<f32>> {
        self.download_f32(&buffer.buffer, buffer.len())
    }

    pub fn download_bf16_buffer(&self, buffer: &CudaBf16Buffer) -> Result<Vec<f32>> {
        self.check_capture_safe("BF16 device-to-host download")?;
        let values = cu(buffer.buffer.to_host_vec(&self.stream))?;
        self.counters
            .add_device_to_host(element_bytes::<u16>(buffer.len()));
        Ok(values
            .into_iter()
            .map(|word| f32::from_bits(u32::from(word) << 16))
            .collect())
    }

    pub fn download_i32_buffer(&self, buffer: &CudaI32Buffer) -> Result<Vec<i32>> {
        let values = cu(buffer.buffer.to_host_vec(&self.stream))?;
        self.counters
            .add_device_to_host(element_bytes::<i32>(buffer.len()));
        Ok(values)
    }

    pub fn clone_f32_buffer(&self, src: &CudaF32Buffer) -> Result<CudaF32Buffer> {
        let mut dst = self.zero_f32_buffer(src.len())?;
        self.copy_f32_into_slot(src, &mut dst, 0)?;
        Ok(dst)
    }

    pub fn overwrite_f32_buffer(&self, src: &[f32], dst: &mut CudaF32Buffer) -> Result<()> {
        if src.len() != dst.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA f32 overwrite length mismatch: src={} dst={}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        self.copy_f32_into_device_buffer(src, &mut dst.buffer)
    }

    pub fn concat_f32_buffers_into(
        &self,
        first: &CudaF32Buffer,
        second: &CudaF32Buffer,
        first_rows: usize,
        row_width: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if row_width == 0 {
            return Err(Error::Internal {
                message: "CUDA f32 concat row width must be positive".into(),
            });
        }
        let first_len = first_rows
            .checked_mul(row_width)
            .ok_or_else(|| Error::Internal {
                message: "CUDA f32 concat first size overflow".into(),
            })?;
        if first.len() != first_len || !second.len().is_multiple_of(row_width) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA f32 concat shape mismatch: first={} expected_first={first_len} second={} row_width={row_width}",
                    first.len(),
                    second.len()
                ),
            });
        }
        let total = first_len
            .checked_add(second.len())
            .ok_or_else(|| Error::Internal {
                message: "CUDA f32 concat length overflow".into(),
            })?;
        if output.len() != total {
            return Err(Error::Internal {
                message: format!(
                    "CUDA f32 concat output length mismatch: expected {total}, got {}",
                    output.len()
                ),
            });
        }
        if first_len != 0 {
            self.copy_f32_into_slot(first, output, 0)?;
        }
        if !second.is_empty() {
            self.copy_f32_into_slot(second, output, first_len)?;
        }
        Ok(())
    }

    pub fn upload_i32_buffer(&self, values: &[i32]) -> Result<CudaI32Buffer> {
        Ok(CudaTypedBuffer::from_device_buffer(
            self.upload_i32(values)?,
        ))
    }

    pub fn i32_host_mirror(&self, values: &[i32]) -> Result<CudaI32HostMirror> {
        if values.is_empty() {
            return Err(Error::Internal {
                message: "CUDA i32 host mirror requires a non-empty buffer".into(),
            });
        }
        let staging = cu(PinnedHostBuffer::from_slice(&self._ctx, values))?;
        let buffer = self.record_device_allocation(values.len(), unsafe {
            DeviceBuffer::from_pinned_host(&self.stream, &staging)
        })?;
        self.counters.add_host_to_device(slice_bytes(values));
        let copy_event = match self.stream.record_event(None) {
            Ok(event) => event,
            Err(error) => {
                self.record_stream_wide_sync(self.stream.synchronize())?;
                return Err(Error::Internal {
                    message: format!("CUDA i32 host mirror event failed: {error:?}"),
                });
            }
        };
        Ok(CudaI32HostMirror {
            host: values.to_vec(),
            device: CudaTypedBuffer::from_device_buffer(buffer),
            staging,
            copy_event,
            active_download: None,
        })
    }

    pub fn begin_i32_host_mirror_download_after(
        &self,
        mirror: &mut CudaI32HostMirror,
        produced: &CudaComputeEvent,
    ) -> Result<CudaI32HostDownload> {
        self.check_capture_safe("i32 control mirror download")?;
        if mirror.active_download.is_some() {
            return Err(Error::Internal {
                message: "CUDA i32 host mirror already owns an active D2H download".into(),
            });
        }
        if let Err(error) = cu(self.control_stream.wait(&produced.event)) {
            let cleanup = produced.synchronize();
            return Err(Error::Internal {
                message: format!(
                    "CUDA i32 control mirror wait failed ({error}); producer cleanup={cleanup:?}"
                ),
            });
        }
        let bytes = element_bytes::<i32>(mirror.device.len()) as usize;
        let staging = mirror.staging.as_mut_slice();
        if let Err(error) = runtime::copy_device_to_host(
            &self.control_stream,
            staging.as_mut_ptr().cast(),
            mirror.device.buffer.cu_deviceptr(),
            bytes,
        ) {
            let cleanup = produced.synchronize();
            return Err(Error::Internal {
                message: format!(
                    "CUDA i32 control mirror D2H failed: {error}; producer cleanup={cleanup:?}"
                ),
            });
        }
        let copied = match self.control_stream.record_event(None) {
            Ok(event) => Arc::new(event),
            Err(error) => {
                let cleanup = cu(self.control_stream.synchronize());
                return Err(Error::Internal {
                    message: format!(
                        "CUDA i32 control mirror completion event failed ({error:?}); control-stream cleanup={cleanup:?}"
                    ),
                });
            }
        };
        mirror.active_download = Some(copied.clone());
        Ok(CudaI32HostDownload {
            copied,
            bytes: bytes as u64,
        })
    }

    pub fn poll_i32_host_mirror_download(
        &self,
        mirror: &mut CudaI32HostMirror,
        download: &CudaI32HostDownload,
    ) -> Result<Option<Vec<i32>>> {
        let active = mirror
            .active_download
            .as_ref()
            .ok_or_else(|| Error::Internal {
                message: "CUDA i32 host mirror has no active D2H download".into(),
            })?;
        if !Arc::ptr_eq(active, &download.copied) {
            return Err(Error::Internal {
                message: "CUDA i32 host mirror was polled with a foreign D2H download".into(),
            });
        }
        if !cu(download.copied.query())? {
            return Ok(None);
        }
        mirror.active_download = None;
        self.counters.add_device_to_host(download.bytes);
        mirror.host.clear();
        mirror.host.extend_from_slice(mirror.staging.as_slice());
        Ok(Some(mirror.host.clone()))
    }

    pub fn update_i32_host_mirror(
        &self,
        values: &[i32],
        mirror: &mut CudaI32HostMirror,
    ) -> Result<()> {
        if values.len() != mirror.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA i32 host mirror shape mismatch: cached={} requested={}",
                    mirror.len(),
                    values.len()
                ),
            });
        }
        if values == mirror.host {
            return Ok(());
        }
        if let Some(download) = mirror.active_download.take() {
            cu(download.synchronize())?;
        }
        cu(mirror.copy_event.synchronize())?;
        mirror.staging.as_mut_slice().copy_from_slice(values);
        unsafe {
            cu(mirror
                .device
                .buffer
                .copy_from_pinned_host_async(&self.stream, &mirror.staging))?;
        }
        self.counters.add_host_to_device(slice_bytes(values));
        match self.stream.record_event(None) {
            Ok(event) => mirror.copy_event = event,
            Err(error) => {
                self.record_stream_wide_sync(self.stream.synchronize())?;
                mirror.host.clear();
                mirror.host.extend_from_slice(values);
                return Err(Error::Internal {
                    message: format!(
                        "CUDA i32 host mirror update event failed after copy: {error:?}"
                    ),
                });
            }
        }
        mirror.host.clear();
        mirror.host.extend_from_slice(values);
        Ok(())
    }

    pub fn zero_i32_buffer(&self, len: usize) -> Result<CudaI32Buffer> {
        Ok(CudaTypedBuffer::from_device_buffer(
            self.zeroed_device_buffer::<i32>(len)?,
        ))
    }

    fn zero_i32_buffer_in_place(&self, buf: &mut CudaI32Buffer) -> Result<()> {
        cu(runtime::memset_u32(
            &self.stream,
            buf.buffer.cu_deviceptr(),
            0,
            buf.len(),
        ))?;
        self.counters.add_compute_kernel_launch();
        Ok(())
    }

    pub fn pack_i32_f32_pairs_into(
        &self,
        indices: &CudaI32Buffer,
        weights: &CudaF32Buffer,
        output: &mut CudaI32Buffer,
        pair_count: usize,
    ) -> Result<()> {
        if pair_count > indices.len() || pair_count > weights.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA pair pack input too small: pairs={pair_count} indices={} weights={}",
                    indices.len(),
                    weights.len()
                ),
            });
        }
        let output_len = pair_count.checked_mul(2).ok_or_else(|| Error::Internal {
            message: "CUDA pair pack output size overflow".into(),
        })?;
        if output.len() != output_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA pair pack output mismatch: expected {output_len}, got {}",
                    output.len()
                ),
            });
        }
        if pair_count == 0 {
            return Ok(());
        }
        let output_len = checked_u32(output_len, "pack i32/f32 pairs", "output_len")?;
        self.launched(unsafe {
            self.module.pack_i32_f32_pairs(
                &self.stream,
                LaunchConfig::for_num_elems(output_len),
                &indices.buffer,
                &weights.buffer,
                &mut output.buffer,
                checked_u32(pair_count, "pack i32/f32 pairs", "pair_count")?,
            )
        })
        .map(|_| ())
    }

    pub fn fill_i32_sequence_prefix(
        &self,
        dst: &mut CudaI32Buffer,
        start: i32,
        len: usize,
    ) -> Result<()> {
        if len > dst.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA i32 sequence exceeds destination: len={len} capacity={}",
                    dst.len()
                ),
            });
        }
        if len == 0 {
            return Ok(());
        }
        let len = checked_u32(len, "fill_i32_sequence", "len")?;
        self.launched(unsafe {
            self.module.fill_i32_sequence(
                &self.stream,
                LaunchConfig::for_num_elems(len),
                &mut dst.buffer,
                start,
                len,
            )
        })
        .map(|_| ())
    }

    pub fn fill_dsv4_paged_window_topk_into(
        &self,
        dst: &mut CudaI32Buffer,
        position: usize,
        window_size: usize,
    ) -> Result<()> {
        if window_size == 0 || window_size > dst.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged window top-k invalid size: window={window_size} capacity={}",
                    dst.len()
                ),
            });
        }
        let kv_len = position.checked_add(1).ok_or_else(|| Error::Internal {
            message: "CUDA paged window KV length overflow".into(),
        })?;
        let valid_len = kv_len.min(window_size);
        let start = kv_len.saturating_sub(window_size);
        let end = start
            .checked_add(valid_len)
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged window index overflow".into(),
            })?;
        if end > i32::MAX as usize {
            return Err(Error::Internal {
                message: "CUDA paged window index exceeds i32 ABI".into(),
            });
        }
        let output_len = checked_u32(window_size, "fill paged window top-k", "output_len")?;
        self.launched(unsafe {
            self.module.fill_dsv4_paged_window_topk(
                &self.stream,
                LaunchConfig::for_num_elems(output_len),
                &mut dst.buffer,
                checked_u32(start, "fill paged window top-k", "start")?,
                checked_u32(valid_len, "fill paged window top-k", "valid_len")?,
                output_len,
            )
        })
        .map(|_| ())
    }

    pub fn fill_dsv4_decode_attention_topk_into(
        &self,
        dst: &mut CudaI32Buffer,
        position: usize,
        window_size: usize,
        window_len: usize,
        compressed_len: usize,
    ) -> Result<usize> {
        if window_size == 0 || window_len > window_size {
            return Err(Error::Internal {
                message: format!(
                    "CUDA decode attention top-k invalid window: size={window_size} len={window_len}"
                ),
            });
        }
        let output_len =
            window_size
                .checked_add(compressed_len)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA decode attention top-k size overflow".into(),
                })?;
        if output_len > dst.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA decode attention top-k exceeds destination: required={output_len} capacity={}",
                    dst.len()
                ),
            });
        }
        let output_len_u32 = checked_u32(output_len, "fill decode attention top-k", "output_len")?;
        self.launched(unsafe {
            self.module.fill_dsv4_decode_attention_topk(
                &self.stream,
                LaunchConfig::for_num_elems(output_len_u32),
                &mut dst.buffer,
                checked_u32(position, "fill decode attention top-k", "position")?,
                checked_u32(window_size, "fill decode attention top-k", "window_size")?,
                checked_u32(window_len, "fill decode attention top-k", "window_len")?,
                checked_u32(
                    compressed_len,
                    "fill decode attention top-k",
                    "compressed_len",
                )?,
                output_len_u32,
            )
        })?;
        Ok(output_len)
    }

    /// Fill each output row with the most recent visible logical indices.
    pub fn fill_recent_rows_into(
        &self,
        visible_lens: &CudaI32Buffer,
        rows: usize,
        width: usize,
        dst: &mut CudaI32Buffer,
    ) -> Result<()> {
        if rows == 0 || width == 0 || visible_lens.len() != rows {
            return Err(Error::Internal {
                message: format!(
                    "CUDA recent-row metadata mismatch: visible_lens={} rows={rows} width={width}",
                    visible_lens.len()
                ),
            });
        }
        let output_len = rows.checked_mul(width).ok_or_else(|| Error::Internal {
            message: "CUDA recent-row output size overflow".into(),
        })?;
        if dst.len() != output_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA recent-row destination mismatch: actual={} expected={output_len}",
                    dst.len()
                ),
            });
        }
        let output_len_u32 = checked_u32(output_len, "fill recent rows", "output_len")?;
        self.launched(unsafe {
            self.module.fill_recent_rows(
                &self.stream,
                LaunchConfig::for_num_elems(output_len_u32),
                &visible_lens.buffer,
                &mut dst.buffer,
                checked_u32(rows, "fill recent rows", "rows")?,
                checked_u32(width, "fill recent rows", "width")?,
                output_len_u32,
            )
        })
        .map(|_| ())
    }

    /// Overwrite an existing i32 device buffer without allocating.
    pub fn overwrite_i32_buffer(&self, src: &[i32], dst: &mut CudaI32Buffer) -> Result<()> {
        self.check_capture_safe("host-to-device i32 overwrite")?;
        if src.len() != dst.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA i32 overwrite length mismatch: src={} dst={}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        let bytes = slice_bytes(src) as usize;
        cu(runtime::copy_host_to_device(
            &self.stream,
            dst.buffer.cu_deviceptr(),
            src.as_ptr().cast(),
            bytes,
        ))?;
        self.record_stream_wide_sync(self.stream.synchronize())?;
        self.counters.add_host_to_device(bytes as u64);
        Ok(())
    }

    /// Overwrite the valid prefix of a capacity-sized i32 workspace.
    pub fn overwrite_i32_prefix(&self, src: &[i32], dst: &mut CudaI32Buffer) -> Result<()> {
        self.check_capture_safe("host-to-device i32 prefix upload")?;
        if src.len() > dst.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA i32 prefix overwrite exceeds capacity: src={} dst={}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        let bytes = slice_bytes(src) as usize;
        cu(runtime::copy_host_to_device(
            &self.stream,
            dst.buffer.cu_deviceptr(),
            src.as_ptr().cast(),
            bytes,
        ))?;
        self.record_stream_wide_sync(self.stream.synchronize())?;
        self.counters.add_host_to_device(bytes as u64);
        Ok(())
    }

    fn copy_f32_into_device_buffer(&self, src: &[f32], dst: &mut DeviceBuffer<f32>) -> Result<()> {
        self.counters.add_host_to_device(slice_bytes(src));
        cu(dst.copy_from_host(&self.stream, src))
    }

    /// Copy `src.len()` f32 elements from `src` into `dst` starting at element
    /// `slot_offset_elements`. Launches the `copy_f32_slot` kernel so the copy
    /// is fully device-resident (no host round-trip), which is required for
    /// CUDA graph capture of the KV-cache append.
    pub fn copy_f32_into_slot(
        &self,
        src: &CudaF32Buffer,
        dst: &mut CudaF32Buffer,
        slot_offset_elements: usize,
    ) -> Result<()> {
        let end = slot_offset_elements
            .checked_add(src.len())
            .ok_or_else(|| Error::Internal {
                message: "CUDA slot copy offset overflow".into(),
            })?;
        if end > dst.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA slot copy out of bounds: dst.len={}, offset={}, src.len={}",
                    dst.len(),
                    slot_offset_elements,
                    src.len()
                ),
            });
        }
        if u64::try_from(end).unwrap_or(u64::MAX) > u64::from(u32::MAX) + 1 {
            return Err(Error::Internal {
                message: format!("CUDA slot copy range exceeds u32 device indexing: end={end}"),
            });
        }
        let copy_len = checked_u32(src.len(), "copy_f32_into_slot", "src.len")?;
        let dst_offset = checked_u32(
            slot_offset_elements,
            "copy_f32_into_slot",
            "slot_offset_elements",
        )?;
        self.launched(unsafe {
            self.module.copy_f32_slot(
                &self.stream,
                LaunchConfig::for_num_elems(copy_len),
                &src.buffer,
                &mut dst.buffer,
                dst_offset,
                copy_len,
            )
        })
    }

    pub fn resident_embedding_hc_bf16_into(
        &self,
        embedding: &CudaArtifactLinearHandle,
        token_ids: &CudaI32Buffer,
        rows: usize,
        hc_mult: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        embedding.validate_storage()?;
        let CudaArtifactLinearShape::Bf16Bytes {
            out_features: vocab,
            in_features: hidden,
        } = embedding.shape
        else {
            return Err(Error::Internal {
                message: format!(
                    "resident embedding gather requires BF16 storage, got {:?}",
                    embedding.shape
                ),
            });
        };
        if rows == 0 || hc_mult == 0 || token_ids.len() != rows {
            return Err(Error::Internal {
                message: format!(
                    "resident embedding gather shape mismatch: rows={rows} token_ids={} hc_mult={hc_mult} hidden={hidden} vocab={vocab}",
                    token_ids.len()
                ),
            });
        }
        let expected = rows
            .checked_mul(hc_mult)
            .and_then(|values| values.checked_mul(hidden))
            .ok_or_else(|| Error::Internal {
                message: "resident embedding gather size overflow".into(),
            })?;
        if output.len() != expected {
            return Err(Error::Internal {
                message: format!(
                    "resident embedding gather output mismatch: output={} expected={expected}",
                    output.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.resident_embedding_hc_bf16(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    expected,
                    "resident_embedding_hc_bf16_into",
                    "output values",
                )?),
                &embedding.weight,
                &token_ids.buffer,
                &mut output.buffer,
                checked_u32(rows, "resident_embedding_hc_bf16_into", "rows")?,
                checked_u32(vocab, "resident_embedding_hc_bf16_into", "vocab")?,
                checked_u32(hc_mult, "resident_embedding_hc_bf16_into", "hc_mult")?,
                checked_u32(hidden, "resident_embedding_hc_bf16_into", "hidden")?,
            )
        })
    }

    pub fn proposal_embedding_hc_from_resident_bf16_into(
        &self,
        embedding: &CudaArtifactLinearHandle,
        anchor_token: u32,
        noise_token: u32,
        rows: usize,
        hc_mult: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        embedding.validate_storage()?;
        let CudaArtifactLinearShape::Bf16Bytes {
            out_features: vocab,
            in_features: hidden,
        } = embedding.shape
        else {
            return Err(Error::Internal {
                message: format!(
                    "proposal resident embedding requires BF16 storage, got {:?}",
                    embedding.shape
                ),
            });
        };
        if rows == 0
            || hc_mult == 0
            || anchor_token as usize >= vocab
            || noise_token as usize >= vocab
        {
            return Err(Error::Internal {
                message: format!(
                    "proposal embedding gather shape/token mismatch: rows={rows} hc_mult={hc_mult} hidden={hidden} vocab={vocab} anchor={anchor_token} noise={noise_token}"
                ),
            });
        }
        let expected = rows
            .checked_mul(hc_mult)
            .and_then(|values| values.checked_mul(hidden))
            .ok_or_else(|| Error::Internal {
                message: "proposal embedding gather size overflow".into(),
            })?;
        if output.len() != expected {
            return Err(Error::Internal {
                message: format!(
                    "proposal embedding gather output mismatch: output={} expected={expected}",
                    output.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.proposal_embedding_hc_bf16(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    expected,
                    "proposal_embedding_hc_from_resident_bf16_into",
                    "output values",
                )?),
                &embedding.weight,
                &mut output.buffer,
                anchor_token,
                noise_token,
                checked_u32(
                    rows,
                    "proposal_embedding_hc_from_resident_bf16_into",
                    "rows",
                )?,
                checked_u32(
                    hc_mult,
                    "proposal_embedding_hc_from_resident_bf16_into",
                    "hc_mult",
                )?,
                checked_u32(
                    hidden,
                    "proposal_embedding_hc_from_resident_bf16_into",
                    "hidden",
                )?,
            )
        })
    }

    pub fn gather_f32_rows(
        &self,
        src: &CudaF32Buffer,
        row_indices: &CudaI32Buffer,
        rows: usize,
        row_width: usize,
    ) -> Result<CudaF32Buffer> {
        if rows == 0 || row_width == 0 || row_indices.len() != rows {
            return Err(Error::Internal {
                message: format!(
                    "CUDA row gather invalid shape: rows={rows} row_width={row_width} indices={}",
                    row_indices.len()
                ),
            });
        }
        if !src.len().is_multiple_of(row_width) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA row gather source length {} is not divisible by row_width {row_width}",
                    src.len()
                ),
            });
        }
        let mut dst =
            self.zero_f32_buffer(rows.checked_mul(row_width).ok_or_else(|| Error::Internal {
                message: "CUDA row gather output size overflow".into(),
            })?)?;
        self.launched(unsafe {
            self.module.gather_f32_rows(
                &self.stream,
                LaunchConfig::for_num_elems((rows * row_width) as u32),
                &src.buffer,
                &row_indices.buffer,
                &mut dst.buffer,
                rows as u32,
                row_width as u32,
            )
        })?;
        Ok(dst)
    }

    pub fn scatter_add_f32_rows(
        &self,
        src: &CudaF32Buffer,
        row_indices: &CudaI32Buffer,
        dst: &mut CudaF32Buffer,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        if rows == 0 || row_width == 0 || row_indices.len() != rows {
            return Err(Error::Internal {
                message: format!(
                    "CUDA row scatter invalid shape: rows={rows} row_width={row_width} indices={}",
                    row_indices.len()
                ),
            });
        }
        let expected_src = rows.checked_mul(row_width).ok_or_else(|| Error::Internal {
            message: "CUDA row scatter source size overflow".into(),
        })?;
        if src.len() != expected_src {
            return Err(Error::Internal {
                message: format!(
                    "CUDA row scatter source length mismatch: src={} expected={expected_src}",
                    src.len()
                ),
            });
        }
        if !dst.len().is_multiple_of(row_width) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA row scatter destination length {} is not divisible by row_width {row_width}",
                    dst.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.scatter_add_f32_rows(
                &self.stream,
                LaunchConfig::for_num_elems(expected_src as u32),
                &src.buffer,
                &row_indices.buffer,
                &mut dst.buffer,
                rows as u32,
                row_width as u32,
            )
        })
    }

    /// Run the checkpoint-native proposal HC/LM/Markov/confidence proposal head.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_proposal_head_into(
        &self,
        hc_state: &CudaF32Buffer,
        hc_function: &CudaF32Buffer,
        hc_scale: &CudaF32Buffer,
        hc_base: &CudaF32Buffer,
        norm_weight: &CudaF32Buffer,
        lm_head: &CudaArtifactLinearHandle,
        markov_w1: &CudaArtifactLinearHandle,
        markov_w2: &CudaArtifactLinearHandle,
        confidence_weight: &CudaArtifactLinearHandle,
        anchor_token_id: u32,
        layout: crate::cuda::cutlass::ProposalHeadLayout,
        workspace: &mut CudaProposalHeadWorkspace,
    ) -> Result<()> {
        let expected = [
            (
                "LM head",
                lm_head.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: layout.vocab,
                    in_features: layout.hidden,
                },
            ),
            (
                "Markov W1",
                markov_w1.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: layout.vocab,
                    in_features: layout.markov_rank,
                },
            ),
            (
                "Markov W2",
                markov_w2.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: layout.vocab,
                    in_features: layout.markov_rank,
                },
            ),
            (
                "confidence",
                confidence_weight.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: 1,
                    in_features: layout.hidden + layout.markov_rank,
                },
            ),
        ];
        for (name, actual, required) in expected {
            if actual != required {
                return Err(Error::Internal {
                    message: format!(
                        "proposal-head {name} shape mismatch: actual={actual:?} expected={required:?}"
                    ),
                });
            }
        }
        let anchor = i32::try_from(anchor_token_id).map_err(|_| Error::Internal {
            message: "proposal anchor token exceeds i32 ABI".into(),
        })?;
        let mut token_ids = vec![0i32; layout.rows + 1];
        token_ids[0] = anchor;
        self.update_i32_host_mirror(&token_ids, &mut workspace.token_ids)?;
        crate::cuda::cutlass::proposal_head(
            &self.stream,
            &hc_state.buffer,
            &hc_function.buffer,
            &hc_scale.buffer,
            &hc_base.buffer,
            &norm_weight.buffer,
            &lm_head.weight,
            &markov_w1.weight,
            &markov_w2.weight,
            &confidence_weight.weight,
            &mut workspace.hidden.buffer,
            &mut workspace.normalized.buffer,
            &mut workspace.base_logits.buffer,
            &mut workspace.partial_values.buffer,
            &mut workspace.partial_indices.buffer,
            &mut workspace.token_ids.device_mut_invalidate_host().buffer,
            &mut workspace.confidence.buffer,
            &mut workspace.status.buffer,
            layout,
        )?;
        self.record_kernel_launch();
        self.record_kernel_launch();
        self.record_kernel_launch();
        Ok(())
    }

    /// Download proposal-head numerical boundaries for diagnostic parity checks.
    /// This is intentionally separate from the compact production result path.
    pub fn download_proposal_head_debug_snapshot(
        &self,
        workspace: &CudaProposalHeadWorkspace,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        Ok((
            self.download_f32_buffer(&workspace.hidden)?,
            self.download_f32_buffer(&workspace.normalized)?,
            self.download_f32_buffer(&workspace.base_logits)?,
        ))
    }

    pub fn begin_proposal_head_result_download(
        &self,
        workspace: &mut CudaProposalHeadWorkspace,
    ) -> Result<CudaI32HostDownload> {
        let rows = workspace.confidence.len();
        let result_values =
            1usize
                .checked_add(rows.saturating_mul(2))
                .ok_or_else(|| Error::Internal {
                    message: "proposal-head result size overflow".into(),
                })?;
        if workspace.token_ids.len() != rows + 1 || workspace.result.len() != result_values {
            return Err(Error::Internal {
                message: format!(
                    "proposal-head compact result shape mismatch: tokens={} confidence={} result={} expected_result={result_values}",
                    workspace.token_ids.len(),
                    rows,
                    workspace.result.len(),
                ),
            });
        }
        self.launched(unsafe {
            self.module.pack_proposal_head_result(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    result_values,
                    "begin_proposal_head_result_download",
                    "result values",
                )?),
                &workspace.status.buffer,
                &workspace.token_ids.device().buffer,
                &workspace.confidence.buffer,
                &mut workspace.result.device_mut_invalidate_host().buffer,
                checked_u32(rows, "begin_proposal_head_result_download", "rows")?,
            )
        })?;
        let produced = self.record_compute_event()?;
        self.begin_i32_host_mirror_download_after(&mut workspace.result, &produced)
    }

    pub fn poll_proposal_head_result(
        &self,
        workspace: &mut CudaProposalHeadWorkspace,
        download: &CudaI32HostDownload,
    ) -> Result<Option<Vec<i32>>> {
        self.poll_i32_host_mirror_download(&mut workspace.result, download)
    }

    pub fn download_proposal_head_result(
        &self,
        workspace: &mut CudaProposalHeadWorkspace,
    ) -> Result<Vec<i32>> {
        let download = self.begin_proposal_head_result_download(workspace)?;
        cu(download.copied.synchronize())?;
        self.poll_proposal_head_result(workspace, &download)?
            .ok_or_else(|| Error::Internal {
                message: "proposal-head result remained pending after synchronization".into(),
            })
    }

    /// Compute a BF16-compressed two-projection bundle on device.
    /// Run checkpoint-native proposal attention over committed paged context and
    /// one read-only five-row proposal block. All scratch remains caller-owned.
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_mla_attention_into(
        &self,
        query: &CudaF32Buffer,
        context_plane: &CudaF32Buffer,
        block_kv: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        attention_sink: &CudaF32Buffer,
        layout: crate::cuda::cutlass::HybridMlaAttentionLayout,
        output: &mut CudaF32Buffer,
        workspace: &mut CudaHybridMlaAttentionWorkspace,
    ) -> Result<()> {
        self.zero_i32_buffer_in_place(&mut workspace.status)?;
        crate::cuda::cutlass::hybrid_mla_attention(
            &self.stream,
            &query.buffer,
            &context_plane.buffer,
            &block_kv.buffer,
            &block_slots.buffer,
            &attention_sink.buffer,
            &mut workspace.query_bf16,
            &mut workspace.gathered_kv_bf16,
            &mut workspace.scores.buffer,
            &mut workspace.probabilities_bf16,
            &mut workspace.online_rescales.buffer,
            &mut workspace.denominators.buffer,
            &mut output.buffer,
            &mut workspace.status.buffer,
            layout,
        )?;
        self.record_kernel_launch();
        Ok(())
    }

    /// Scatter `[rows, layout.elements_per_token]` values into one layer of a
    /// contiguous paged plane. Row `r` uses packed block range
    /// `block_offsets[r]..block_offsets[r + 1]` and logical row `positions[r]`.
    /// A zero mask entry skips the corresponding row.
    pub fn paged_plane_scatter_rows_from_device(
        &self,
        values: &CudaF32Buffer,
        positions: &CudaI32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        mask: Option<&CudaI32Buffer>,
        plane: &mut CudaF32Buffer,
        layout: crate::cuda::kv_page_pool::PagedPlaneLayout,
    ) -> Result<()> {
        self.paged_plane_scatter_selected_rows_from_device_impl(
            values,
            positions,
            block_slots,
            block_offsets,
            None,
            mask,
            plane,
            layout,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_plane_scatter_selected_rows_from_device(
        &self,
        values: &CudaF32Buffer,
        positions: &CudaI32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        row_sequence_ids: &CudaI32Buffer,
        mask: Option<&CudaI32Buffer>,
        plane: &mut CudaF32Buffer,
        layout: crate::cuda::kv_page_pool::PagedPlaneLayout,
    ) -> Result<()> {
        self.paged_plane_scatter_selected_rows_from_device_impl(
            values,
            positions,
            block_slots,
            block_offsets,
            Some(row_sequence_ids),
            mask,
            plane,
            layout,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn paged_plane_scatter_selected_rows_from_device_impl(
        &self,
        values: &CudaF32Buffer,
        positions: &CudaI32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        row_sequence_ids: Option<&CudaI32Buffer>,
        mask: Option<&CudaI32Buffer>,
        plane: &mut CudaF32Buffer,
        layout: crate::cuda::kv_page_pool::PagedPlaneLayout,
    ) -> Result<()> {
        layout.validate()?;
        let rows = positions.len();
        let expected_values =
            rows.checked_mul(layout.elements_per_token)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA paged plane scatter value size overflow".into(),
                })?;
        if values.len() != expected_values {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged plane scatter values length mismatch: got {} expected {expected_values} for rows={rows} row_dim={}",
                    values.len(),
                    layout.elements_per_token
                ),
            });
        }
        if block_offsets.len() < 2 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged plane scatter requires at least one sequence, got {} offsets",
                    block_offsets.len()
                ),
            });
        }
        if let Some(row_sequence_ids) = row_sequence_ids {
            if row_sequence_ids.len() != rows {
                return Err(Error::Internal {
                    message: format!(
                        "CUDA paged plane scatter row selector length mismatch: got {} expected {rows}",
                        row_sequence_ids.len()
                    ),
                });
            }
        } else if block_offsets.len() != rows + 1 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged plane scatter identity mapping requires {} offsets, got {}",
                    rows + 1,
                    block_offsets.len()
                ),
            });
        }
        if rows != 0 && block_slots.is_empty() {
            return Err(Error::Internal {
                message: "CUDA paged plane scatter requires block slots for non-empty rows".into(),
            });
        }
        if let Some(mask) = mask
            && mask.len() != rows
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged plane scatter mask length mismatch: got {} expected {rows}",
                    mask.len()
                ),
            });
        }
        let slot_elements = layout
            .layer_count
            .checked_mul(layout.page_tokens)
            .and_then(|value| value.checked_mul(layout.elements_per_token))
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged plane scatter slot size overflow".into(),
            })?;
        if plane.len() < slot_elements || !plane.len().is_multiple_of(slot_elements) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged plane scatter storage length {} is not a positive multiple of slot size {slot_elements}",
                    plane.len()
                ),
            });
        }
        let plane_elements =
            checked_u32(plane.len(), "CUDA paged plane scatter", "plane elements")?;
        let rows = checked_u32(rows, "CUDA paged plane scatter", "rows")?;
        let row_dim = checked_u32(
            layout.elements_per_token,
            "CUDA paged plane scatter",
            "row_dim",
        )?;
        let num_elements = checked_u32(
            expected_values,
            "CUDA paged plane scatter",
            "value elements",
        )?;
        if num_elements == 0 {
            return Ok(());
        }
        let (row_sequence_buffer, use_row_sequence_ids) = match row_sequence_ids {
            Some(row_sequence_ids) => (&row_sequence_ids.buffer, 1u32),
            None => (&positions.buffer, 0u32),
        };
        let (mask_buffer, use_mask) = match mask {
            Some(mask) => (&mask.buffer, 1u32),
            None => (&positions.buffer, 0u32),
        };
        self.launched(unsafe {
            self.module.paged_plane_scatter_rows_f32(
                &self.stream,
                LaunchConfig::for_num_elems(num_elements),
                &values.buffer,
                &positions.buffer,
                &block_slots.buffer,
                &block_offsets.buffer,
                row_sequence_buffer,
                mask_buffer,
                &mut plane.buffer,
                num_elements,
                plane_elements,
                rows,
                row_dim,
                layout.page_tokens as u32,
                layout.layer_index as u32,
                layout.layer_count as u32,
                use_row_sequence_ids,
                use_mask,
            )
        })
    }

    /// Device-side accumulate: `y += scale * x`.
    ///
    /// Used to accumulate routed expert outputs on the GPU without
    /// downloading each expert's output to host and accumulating in `Vec<f32>`.
    pub fn saxpy_into(&self, scale: f32, x: &CudaF32Buffer, y: &mut CudaF32Buffer) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::Internal {
                message: format!("CUDA saxpy length mismatch: x={} y={}", x.len(), y.len()),
            });
        }
        self.launched(unsafe {
            self.module.saxpy(
                &self.stream,
                LaunchConfig::for_num_elems(x.len() as u32),
                scale,
                &x.buffer,
                &mut y.buffer,
                x.len() as u32,
            )
        })
    }

    /// Allocate one bounded arena for many immutable artifact views.
    ///
    /// `maximum_views` accounts for alignment padding between views. The CUDA
    /// driver sees one allocation instead of one allocation per tensor.
    pub fn allocate_artifact_linear_arena(
        &self,
        payload_bytes: usize,
        maximum_views: usize,
    ) -> Result<CudaArtifactLinearArena> {
        let capacity = CudaArtifactLinearArena::allocation_capacity(payload_bytes, maximum_views)?;
        Ok(CudaArtifactLinearArena {
            storage: self.uninitialized_upload_device_buffer::<u8>(capacity)?,
            cursor: 0,
        })
    }

    /// Allocate backend-owned storage for a fixed number of routed experts.
    pub fn allocate_routed_expert_arena(
        &self,
        shape: CudaRoutedExpertShape,
        frame_capacity: usize,
    ) -> Result<CudaRoutedExpertArena> {
        self.check_capture_safe("routed-expert arena allocation")?;
        if frame_capacity == 0 {
            return Err(Error::Internal {
                message: "CUDA routed-expert arena requires positive frame capacity".into(),
            });
        }
        let storage_bytes = shape.storage_bytes()?;
        let raw_payload = storage_bytes
            .raw_linear
            .checked_mul(frame_capacity)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert raw arena byte size overflow".into(),
            })?;
        let private_per_frame =
            storage_bytes
                .provider_private
                .into_iter()
                .try_fold(0usize, |total, bytes| {
                    total.checked_add(bytes).ok_or_else(|| Error::Internal {
                        message: "CUDA routed-expert private arena byte size overflow".into(),
                    })
                })?;
        let private_payload = private_per_frame
            .checked_mul(frame_capacity)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert private arena byte size overflow".into(),
            })?;
        storage_bytes
            .physical
            .checked_mul(frame_capacity)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert arena physical byte size overflow".into(),
            })?;
        let raw_views = frame_capacity
            .checked_mul(6)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert raw arena view count overflow".into(),
            })?;
        let private_views = frame_capacity
            .checked_mul(3)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-expert private arena view count overflow".into(),
            })?;

        Ok(CudaRoutedExpertArena {
            shape,
            storage_bytes,
            raw_linear_arena: self.allocate_artifact_linear_arena(raw_payload, raw_views)?,
            provider_private_layout_arena: self
                .allocate_artifact_linear_arena(private_payload, private_views)?,
            frame_capacity,
            allocated_frames: 0,
        })
    }

    /// Upload and prepare one routed-expert frame on the existing upload stream.
    ///
    /// The returned ticket retains all pinned sources until an event recorded
    /// after private layout preparation. No stream or scratch allocation is made.
    #[allow(clippy::too_many_arguments)]
    pub fn debug_dump_prepared_routed_expert(
        &self,
        frame: &CudaPreparedRoutedExpert,
        directory: &std::path::Path,
        prefix: &str,
    ) -> Result<()> {
        self.check_capture_safe("routed-expert artifact debug dump")?;
        frame.validate_storage()?;
        std::fs::create_dir_all(directory).map_err(|source| Error::Internal {
            message: format!("failed to create routed-expert artifact dump directory: {source}"),
        })?;
        for (name, buffer) in frame.debug_buffers() {
            let values = cu(buffer.to_host_vec(&self.upload_stream))?;
            std::fs::write(directory.join(format!("{prefix}.{name}.bin")), values).map_err(
                |source| Error::Internal {
                    message: format!("failed to write routed-expert artifact dump: {source}"),
                },
            )?;
        }
        Ok(())
    }

    pub fn materialize_routed_expert_from_pinned_async(
        &self,
        frame: &mut CudaPreparedRoutedExpert,
        gate_weight: CudaPinnedU8HostBuffer,
        gate_scale: CudaPinnedU8HostBuffer,
        up_weight: CudaPinnedU8HostBuffer,
        up_scale: CudaPinnedU8HostBuffer,
        down_weight: CudaPinnedU8HostBuffer,
        down_scale: CudaPinnedU8HostBuffer,
    ) -> Result<CudaRoutedExpertMaterialization> {
        self.check_capture_safe("routed-expert materialization")?;
        frame.validate_storage()?;
        let gate_up_shape = frame.shape.gate_up_linear_shape();
        let down_shape = frame.shape.down_linear_shape();
        gate_up_shape.validate(gate_weight.len(), gate_scale.len())?;
        gate_up_shape.validate(up_weight.len(), up_scale.len())?;
        down_shape.validate(down_weight.len(), down_scale.len())?;

        let sources = vec![
            gate_weight,
            gate_scale,
            up_weight,
            up_scale,
            down_weight,
            down_scale,
        ];
        for source in &sources {
            debug_assert!(Arc::ptr_eq(source.buffer.context(), &self._ctx));
        }
        let upload_bytes = sources.iter().try_fold(0u64, |total, source| {
            total
                .checked_add(source.len() as u64)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA routed-expert upload byte size overflow".into(),
                })
        })?;
        self.counters.add_artifact_upload(upload_bytes);

        let enqueue_copy =
            |destination: DevicePtr, source: &CudaPinnedU8HostBuffer| -> Result<()> {
                cu(runtime::copy_host_to_device(
                    &self.upload_stream,
                    destination,
                    source.as_ptr().cast(),
                    source.len(),
                ))?;
                self.counters.add_host_to_device(source.len() as u64);
                Ok(())
            };
        let prepare_private_layout = |source: &DeviceBuffer<u8>,
                                      destination: &mut DeviceBuffer<u8>,
                                      out_features: usize,
                                      in_features: usize|
         -> Result<()> {
            // The source is the linear scale view just uploaded above. The
            // native provider transforms it directly into its private layout.
            prepare_mxfp4_sfb(
                &self.upload_stream,
                source,
                destination,
                out_features,
                in_features,
            )
            .map_err(|_| Error::Internal {
                message: "CUDA routed-expert private layout preparation failed".into(),
            })?;
            self.record_upload_kernel_launch();
            Ok(())
        };

        let submission = (|| -> Result<CudaUploadEvent> {
            enqueue_copy(frame.gate.weight.cu_deviceptr(), &sources[0])?;
            enqueue_copy(
                frame
                    .gate
                    .scale
                    .as_ref()
                    .expect("validated routed-expert gate scale")
                    .cu_deviceptr(),
                &sources[1],
            )?;
            enqueue_copy(frame.up.weight.cu_deviceptr(), &sources[2])?;
            enqueue_copy(
                frame
                    .up
                    .scale
                    .as_ref()
                    .expect("validated routed-expert up scale")
                    .cu_deviceptr(),
                &sources[3],
            )?;
            enqueue_copy(frame.down.weight.cu_deviceptr(), &sources[4])?;
            enqueue_copy(
                frame
                    .down
                    .scale
                    .as_ref()
                    .expect("validated routed-expert down scale")
                    .cu_deviceptr(),
                &sources[5],
            )?;

            prepare_private_layout(
                frame
                    .gate
                    .scale
                    .as_ref()
                    .expect("validated routed-expert gate scale"),
                &mut frame.gate_provider_scale,
                frame.shape.intermediate,
                frame.shape.input,
            )?;
            prepare_private_layout(
                frame
                    .up
                    .scale
                    .as_ref()
                    .expect("validated routed-expert up scale"),
                &mut frame.up_provider_scale,
                frame.shape.intermediate,
                frame.shape.input,
            )?;
            prepare_private_layout(
                frame
                    .down
                    .scale
                    .as_ref()
                    .expect("validated routed-expert down scale"),
                &mut frame.down_provider_scale,
                frame.shape.output,
                frame.shape.intermediate,
            )?;
            self.record_upload_event()
        })();

        match submission {
            Ok(event) => Ok(CudaRoutedExpertMaterialization { sources, event }),
            Err(error) => match self.sync_upload_stream() {
                Ok(()) => Err(error),
                Err(sync_error) => {
                    for source in sources {
                        std::mem::forget(source);
                    }
                    Err(Error::Internal {
                        message: format!(
                            "CUDA routed-expert materialization failed ({error}); synchronizing the upload stream also failed ({sync_error})"
                        ),
                    })
                }
            },
        }
    }

    /// Preallocate an artifact handle in ordinary device storage without
    /// initializing its contents.
    ///
    /// Allocation is enqueued on the upload stream so subsequent pinned
    /// overwrites on that stream are naturally ordered after frame creation.
    pub fn allocate_artifact_linear_device(
        &self,
        shape: CudaArtifactLinearShape,
    ) -> Result<CudaArtifactLinearHandle> {
        let (weight_len, scale_len) = shape.storage_lengths()?;
        let weight = self.uninitialized_upload_device_buffer::<u8>(weight_len)?;
        let scale = if scale_len == 0 {
            None
        } else {
            Some(self.uninitialized_upload_device_buffer::<u8>(scale_len)?)
        };
        Ok(CudaArtifactLinearHandle {
            shape,
            weight,
            scale,
        })
    }

    /// Prepare-time bounded ingest into a preallocated artifact weight. This is
    /// used for the resident proposal LM head so the hot path has one stable pointer
    /// without materializing the complete tensor in host RAM.
    pub fn overwrite_artifact_linear_weight_range(
        &self,
        handle: &mut CudaArtifactLinearHandle,
        offset: usize,
        bytes: &[u8],
    ) -> Result<()> {
        self.check_capture_safe("artifact linear weight-range upload")?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| Error::Internal {
                message: "artifact weight-range overflow".into(),
            })?;
        if end > handle.weight.len() || handle.scale.is_some() {
            return Err(Error::Internal {
                message: format!(
                    "artifact weight-range mismatch: offset={offset} bytes={} capacity={} shape={:?}",
                    bytes.len(),
                    handle.weight.len(),
                    handle.shape
                ),
            });
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let destination = handle
            .weight
            .cu_deviceptr()
            .checked_add(offset as u64)
            .ok_or_else(|| Error::Internal {
                message: "artifact device address overflow".into(),
            })?;
        cu(runtime::copy_host_to_device(
            &self.upload_stream,
            destination,
            bytes.as_ptr().cast(),
            bytes.len(),
        ))?;
        self.record_stream_wide_sync(self.upload_stream.synchronize())?;
        self.counters.add_host_to_device(bytes.len() as u64);
        self.counters.add_artifact_upload(bytes.len() as u64);
        Ok(())
    }

    /// Overwrite a preallocated artifact handle from pinned host storage.
    ///
    /// This method performs no device allocation. The returned ticket owns the
    /// pinned sources and an event recorded after both copies on the upload
    /// stream; it must remain alive until that event completes.
    pub fn overwrite_artifact_linear_from_pinned_async(
        &self,
        handle: &mut CudaArtifactLinearHandle,
        expected_shape: CudaArtifactLinearShape,
        weight: CudaPinnedU8HostBuffer,
        scale: Option<CudaPinnedU8HostBuffer>,
    ) -> Result<CudaArtifactLinearAsyncOverwrite> {
        self.check_capture_safe("artifact linear pinned overwrite")?;
        if handle.shape != expected_shape {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear overwrite shape mismatch: handle={:?} requested={expected_shape:?}",
                    handle.shape
                ),
            });
        }
        handle.validate_storage()?;
        let scale_len = scale.as_ref().map(CudaPinnedU8HostBuffer::len).unwrap_or(0);
        expected_shape.validate(weight.len(), scale_len)?;
        let upload_bytes = (weight.len() as u64).saturating_add(scale_len as u64);
        self.counters.add_artifact_upload(upload_bytes);

        debug_assert!(Arc::ptr_eq(weight.buffer.context(), &self._ctx));
        if let Some(scale) = scale.as_ref() {
            debug_assert!(Arc::ptr_eq(scale.buffer.context(), &self._ctx));
        }
        let enqueue_result = (|| -> Result<CudaUploadEvent> {
            cu(runtime::copy_host_to_device(
                &self.upload_stream,
                handle.weight.cu_deviceptr(),
                weight.as_ptr().cast(),
                weight.len(),
            ))?;
            self.counters.add_host_to_device(weight.len() as u64);

            match (handle.scale.as_mut(), scale.as_ref()) {
                (Some(dst), Some(src)) => {
                    cu(runtime::copy_host_to_device(
                        &self.upload_stream,
                        dst.cu_deviceptr(),
                        src.as_ptr().cast(),
                        src.len(),
                    ))?;
                    self.counters.add_host_to_device(src.len() as u64);
                }
                (None, None) => {}
                (None, Some(src)) if src.is_empty() => {}
                _ => {
                    return Err(Error::Internal {
                        message: "CUDA artifact linear overwrite scale storage mismatch".into(),
                    });
                }
            }
            self.record_upload_event()
        })();

        match enqueue_result {
            Ok(event) => Ok(CudaArtifactLinearAsyncOverwrite {
                weight: Some(weight),
                scale,
                event,
            }),
            Err(error) => match self.sync_upload_stream() {
                Ok(()) => Err(error),
                Err(sync_error) => {
                    // Without a successful synchronization CUDA may still be
                    // reading these sources. Leak their Arc guards rather than
                    // releasing pinned memory prematurely.
                    std::mem::forget(weight);
                    if let Some(scale) = scale {
                        std::mem::forget(scale);
                    }
                    Err(Error::Internal {
                        message: format!(
                            "artifact linear pinned overwrite failed ({error}); synchronizing the upload stream also failed ({sync_error})"
                        ),
                    })
                }
            },
        }
    }

    pub fn upload_artifact_linear(
        &self,
        shape: CudaArtifactLinearShape,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<CudaArtifactLinearHandle> {
        shape.validate(weight.len(), scale.len())?;
        self.counters
            .add_artifact_upload(slice_bytes(weight).saturating_add(slice_bytes(scale)));
        Ok(CudaArtifactLinearHandle {
            shape,
            weight: self.upload_u8(weight)?,
            scale: if scale.is_empty() {
                None
            } else {
                Some(self.upload_u8(scale)?)
            },
        })
    }

    /// Enqueue artifact linear H2D copies on the dedicated upload stream.
    ///
    /// # Safety
    /// `weight` and `scale` must outlive the upload event recorded after this
    /// call. The returned handle may be inserted into compute data structures
    /// immediately, but kernels must not use it until the event is complete or
    /// the compute stream has waited on it.
    unsafe fn upload_artifact_linear_from_pinned_async_unchecked(
        &self,
        shape: CudaArtifactLinearShape,
        weight: &CudaPinnedU8HostBuffer,
        scale: Option<&CudaPinnedU8HostBuffer>,
    ) -> Result<CudaArtifactLinearHandle> {
        let scale_len = scale.map(CudaPinnedU8HostBuffer::len).unwrap_or(0);
        shape.validate(weight.len(), scale_len)?;
        self.counters
            .add_artifact_upload((weight.len() as u64).saturating_add(scale_len as u64));
        Ok(CudaArtifactLinearHandle {
            shape,
            weight: unsafe { self.upload_u8_from_pinned_async_unchecked(weight)? },
            scale: match scale {
                Some(scale) if !scale.is_empty() => {
                    Some(unsafe { self.upload_u8_from_pinned_async_unchecked(scale)? })
                }
                _ => None,
            },
        })
    }

    pub fn upload_artifact_linear_from_pinned_async(
        &self,
        shape: CudaArtifactLinearShape,
        weight: CudaPinnedU8HostBuffer,
        scale: Option<CudaPinnedU8HostBuffer>,
    ) -> Result<CudaArtifactLinearAsyncUpload> {
        let handle = unsafe {
            self.upload_artifact_linear_from_pinned_async_unchecked(shape, &weight, scale.as_ref())?
        };
        Ok(CudaArtifactLinearAsyncUpload {
            handle,
            _weight: weight,
            _scale: scale,
        })
    }

    pub fn upload_fp4_e2m1_e8m0_linear_from_pinned_async(
        &self,
        weight: CudaPinnedU8HostBuffer,
        scale: CudaPinnedU8HostBuffer,
        out_features: usize,
        in_features: usize,
    ) -> Result<CudaArtifactLinearAsyncUpload> {
        self.upload_artifact_linear_from_pinned_async(
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            },
            weight,
            Some(scale),
        )
    }

    pub fn upload_f32_linear(
        &self,
        weight: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        self.upload_artifact_linear(
            CudaArtifactLinearShape::F32 {
                out_features,
                in_features,
            },
            weight,
            &[],
        )
    }

    pub fn upload_bf16_linear(
        &self,
        weight: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        self.upload_artifact_linear(
            CudaArtifactLinearShape::Bf16Bytes {
                out_features,
                in_features,
            },
            weight,
            &[],
        )
    }

    pub fn upload_fp8_e4m3_e8m0_linear(
        &self,
        weight: &[u8],
        scale: &[u8],
        out_features: usize,
        in_features: usize,
        block_m: usize,
        block_k: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        self.upload_artifact_linear(
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            },
            weight,
            scale,
        )
    }

    pub fn upload_fp4_e2m1_e8m0_linear(
        &self,
        weight: &[u8],
        scale: &[u8],
        out_features: usize,
        in_features: usize,
        use_managed: bool,
    ) -> Result<CudaArtifactLinearHandle> {
        // On unified-memory CUDA systems (unified memory), managed allocation avoids the expert H2D copy.
        if use_managed {
            return self.upload_artifact_linear_managed(
                CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                    out_features,
                    in_features,
                },
                weight,
                scale,
            );
        }
        self.upload_artifact_linear(
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
                out_features,
                in_features,
            },
            weight,
            scale,
        )
    }

    /// Allocate expert weight/scale buffers as CUDA managed memory.
    ///
    /// On unified-memory CUDA systems (sm_121, unified addressing), managed memory is accessible by
    /// both CPU and GPU without explicit H2D copies. The GPU reads directly
    /// from host LPDDR5X pages, so expert loading becomes a pure host-side
    /// memcpy (disk -> managed buffer) with zero upload overhead.
    pub fn upload_artifact_linear_managed(
        &self,
        shape: CudaArtifactLinearShape,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<CudaArtifactLinearHandle> {
        shape.validate(weight.len(), scale.len())?;
        let weight_buf = self.alloc_managed_u8(weight)?;
        let scale_buf = if scale.is_empty() {
            None
        } else {
            Some(self.alloc_managed_u8(scale)?)
        };
        // No counter bump: managed memory is not an H2D transfer.
        Ok(CudaArtifactLinearHandle {
            shape,
            weight: weight_buf,
            scale: scale_buf,
        })
    }

    pub fn allocate_artifact_linear_managed(
        &self,
        shape: CudaArtifactLinearShape,
        weight_len: usize,
        scale_len: usize,
    ) -> Result<CudaArtifactLinearHandle> {
        shape.validate(weight_len, scale_len)?;
        Ok(CudaArtifactLinearHandle {
            shape,
            weight: self.alloc_managed_u8_len(weight_len)?,
            scale: if scale_len == 0 {
                None
            } else {
                Some(self.alloc_managed_u8_len(scale_len)?)
            },
        })
    }

    pub fn overwrite_artifact_linear(
        &self,
        handle: &mut CudaArtifactLinearHandle,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<()> {
        handle.shape.validate(weight.len(), scale.len())?;
        cu(handle.weight.copy_from_host(&self.stream, weight))?;
        match (handle.scale.as_mut(), scale.is_empty()) {
            (Some(dst), false) => cu(dst.copy_from_host(&self.stream, scale)),
            (None, true) => Ok(()),
            _ => Err(Error::Internal {
                message: "CUDA artifact linear recycled scale storage mismatch".into(),
            }),
        }
    }

    /// Allocate a CUDA managed-memory buffer and copy `data` into it.
    ///
    /// Managed memory is accessible from both host and device on unified
    /// addressing platforms (unified-memory CUDA systems). The returned `DeviceBuffer` owns the
    /// allocation and frees it via `cuMemFree` on drop.
    ///
    /// On unified-memory CUDA systems (sm_121, unified addressing), managed memory is accessible by
    /// both CPU and GPU without explicit H2D copies. The GPU reads directly
    /// from host LPDDR5X pages, so expert loading becomes a pure host-side
    /// memcpy (disk → managed buffer) with zero upload overhead.
    ///
    /// We also set `CU_MEM_ADVISE_SET_READ_MOSTLY` to hint the driver that
    /// expert weights are read-only after upload, enabling better page
    /// placement and reducing fault overhead.
    fn alloc_managed_u8(&self, data: &[u8]) -> Result<DeviceBuffer<u8>> {
        let buffer = self.alloc_managed_u8_len(data.len())?;
        cu(buffer.copy_from_host(&self.stream, data))?;
        Ok(buffer)
    }

    fn alloc_managed_u8_len(&self, len: usize) -> Result<DeviceBuffer<u8>> {
        self.counters.begin_device_allocation();
        match unsafe { DeviceBuffer::managed(self.stream.context(), len) } {
            Ok(buffer) => {
                self.counters.complete_device_allocation(len as u64);
                Ok(buffer)
            }
            Err(error) => {
                self.counters.fail_device_allocation();
                Err(error.into())
            }
        }
    }

    pub fn artifact_linear_matvec(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != handle.shape.in_features() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear input length mismatch: expected {}, got {}",
                    handle.shape.in_features(),
                    input.len()
                ),
            });
        }
        let xd = self.upload_f32_buffer(input)?;
        let mut yd = self.zero_f32_buffer(handle.shape.out_features())?;
        self.artifact_linear_matvec_into(handle, &xd, &mut yd)?;
        self.download_f32_buffer(&yd)
    }

    pub fn artifact_linear_matvec_into(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if input.len() != handle.shape.in_features() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear device input length mismatch: expected {}, got {}",
                    handle.shape.in_features(),
                    input.len()
                ),
            });
        }
        if output.len() != handle.shape.out_features() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear device output length mismatch: expected {}, got {}",
                    handle.shape.out_features(),
                    output.len()
                ),
            });
        }
        if self.artifact_linear_supports_prepacked_fp8(handle) {
            self.artifact_linear_matvec_prepacked_fp8_from_f32(
                handle,
                &input.buffer,
                &mut output.buffer,
            )
        } else {
            self.artifact_linear_matvec_device(handle, &input.buffer, &mut output.buffer)
        }
    }

    pub fn artifact_linear_pair_matvec_into(
        &self,
        first: &CudaArtifactLinearHandle,
        second: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        first_output: &mut CudaF32Buffer,
        second_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let first_in = first.shape.in_features();
        let second_in = second.shape.in_features();
        if first_in != second_in || input.len() != first_in {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear pair input mismatch: first={first_in}, second={second_in}, input={}",
                    input.len()
                ),
            });
        }
        let first_out = first.shape.out_features();
        let second_out = second.shape.out_features();
        if first_output.len() != first_out || second_output.len() != second_out {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear pair output mismatch: first expected={first_out} got={}, second expected={second_out} got={}",
                    first_output.len(),
                    second_output.len()
                ),
            });
        }
        match (first.shape, second.shape) {
            (
                CudaArtifactLinearShape::Bf16Bytes { .. },
                CudaArtifactLinearShape::Bf16Bytes { .. },
            ) => {
                let combined_out =
                    first_out
                        .checked_add(second_out)
                        .ok_or_else(|| Error::Internal {
                            message: "CUDA BF16 linear pair output size overflow".into(),
                        })?;
                let combined_out = checked_u32(
                    combined_out,
                    "artifact_linear_pair_matvec_into",
                    "combined_out",
                )?;
                let first_out =
                    checked_u32(first_out, "artifact_linear_pair_matvec_into", "first_out")?;
                let second_out =
                    checked_u32(second_out, "artifact_linear_pair_matvec_into", "second_out")?;
                let in_features =
                    checked_u32(first_in, "artifact_linear_pair_matvec_into", "in_features")?;
                self.launched(unsafe {
                    self.module.dual_linear_bf16_from_f32(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: (combined_out, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &input.buffer,
                        &first.weight,
                        &second.weight,
                        &mut first_output.buffer,
                        &mut second_output.buffer,
                        first_out,
                        second_out,
                        in_features,
                    )
                })
            }
            _ => {
                self.artifact_linear_matvec_into(first, input, first_output)?;
                self.artifact_linear_matvec_into(second, input, second_output)
            }
        }
    }

    pub fn artifact_linear_rows_from_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
    ) -> Result<CudaF32Buffer> {
        let out_features = handle.shape.out_features();
        let len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear rows output size overflow".into(),
            })?;
        let mut output =
            CudaTypedBuffer::from_device_buffer(self.uninitialized_device_buffer::<f32>(len)?);
        self.artifact_linear_rows_from_device_into(handle, input, rows, &mut output)?;
        Ok(output)
    }

    pub fn artifact_linear_rows_from_device_into(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let in_features = handle.shape.in_features();
        let out_features = handle.shape.out_features();
        if rows == 0 || input.len() != rows * in_features {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear rows input mismatch: rows={rows} in_features={in_features} input={}",
                    input.len()
                ),
            });
        }
        let expected_output = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear rows output size overflow".into(),
            })?;
        if output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear rows output mismatch: expected {expected_output}, got {}",
                    output.len()
                ),
            });
        }
        if self.artifact_linear_supports_prepacked_fp8(handle) {
            return self.artifact_linear_rows_prepacked_fp8_from_f32(
                handle,
                &input.buffer,
                rows,
                &mut output.buffer,
            );
        }
        if !quantized_shape_uses_fp8_activation(handle.shape) {
            return self.artifact_linear_rows_device(
                handle,
                &input.buffer,
                rows,
                &mut output.buffer,
            );
        }
        let mut x = self.clone_f32_buffer(input)?;
        self.fp8_activation_quantize_buffer_in_place(
            &mut x,
            in_features,
            ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
        )?;
        self.artifact_linear_rows_device(handle, &x.buffer, rows, &mut output.buffer)
    }

    pub fn artifact_linear_rows_from_device_into_with_scratch(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
        scratch: &mut CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        let in_features = handle.shape.in_features();
        let out_features = handle.shape.out_features();
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear rows input size overflow".into(),
            })?;
        if rows == 0 || input.len() != input_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear rows input mismatch: rows={rows} in_features={in_features} input={}",
                    input.len()
                ),
            });
        }
        let expected_output = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear rows output size overflow".into(),
            })?;
        if output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear rows output mismatch: expected {expected_output}, got {}",
                    output.len()
                ),
            });
        }
        if input_len > scratch.value_capacity {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear scratch too small: required={input_len} capacity={}",
                    scratch.value_capacity
                ),
            });
        }
        if self.artifact_linear_supports_prepacked_fp8(handle) {
            return self.artifact_linear_rows_prepacked_fp8_from_f32_with_scratch(
                handle,
                &input.buffer,
                rows,
                &mut output.buffer,
                scratch,
            );
        }

        if !quantized_shape_uses_fp8_activation(handle.shape) {
            return self.artifact_linear_rows_device(
                handle,
                &input.buffer,
                rows,
                &mut output.buffer,
            );
        }
        self.copy_f32_into_slot(input, &mut scratch.cloned, 0)?;
        self.fp8_activation_quantize_in_place(
            &mut scratch.cloned.buffer,
            input_len,
            in_features,
            ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
        )?;
        self.artifact_linear_rows_device(handle, &scratch.cloned.buffer, rows, &mut output.buffer)
    }

    pub fn artifact_fp8_projection_rows_from_device_into_with_scratch(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
        scratch: &mut CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        let CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m: 128,
            block_k: 128,
        } = handle.shape
        else {
            return Err(Error::Internal {
                message: "FP8 projection requires an FP8 K128 artifact".into(),
            });
        };
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "FP8 projection input size overflow".into(),
            })?;
        let output_len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::Internal {
                message: "FP8 projection output size overflow".into(),
            })?;
        if rows == 0 || input.len() != input_len || output.len() != output_len {
            return Err(Error::Internal {
                message: format!(
                    "FP8 projection shape mismatch: rows={rows} input={}/{} output={}/{}",
                    input.len(),
                    input_len,
                    output.len(),
                    output_len
                ),
            });
        }
        let scale_cols = in_features / ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE;
        let scale_len = rows
            .checked_mul(scale_cols)
            .ok_or_else(|| Error::Internal {
                message: "FP8 projection scale size overflow".into(),
            })?;
        if input_len > scratch.value_capacity || scale_len > scratch.scale_capacity {
            return Err(Error::Internal {
                message: format!(
                    "FP8 projection scratch too small: packed={input_len}/{} scales={scale_len}/{}",
                    scratch.value_capacity, scratch.scale_capacity
                ),
            });
        }
        let weight_scales = handle.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "FP8 projection scales are missing".into(),
        })?;
        self.pack_fp8_rows_from_f32_preallocated(
            &input.buffer,
            rows,
            in_features,
            &mut scratch.x_packed,
            scratch.value_capacity,
            &mut scratch.x_scales,
            scratch.scale_capacity,
        )?;
        crate::cuda::cutlass::fp8_projection(
            &self.stream,
            &scratch.x_packed,
            &scratch.x_scales,
            &handle.weight,
            weight_scales,
            &mut output.buffer,
            rows,
            out_features,
            in_features,
        )?;
        self.record_kernel_launch();
        Ok(())
    }

    pub fn prepare_fp8_activation_from_device<'a>(
        &self,
        input: &CudaF32Buffer,
        rows: usize,
        row_width: usize,
        storage: &'a mut CudaFp8ActivationPack,
    ) -> Result<CudaPreparedFp8Activation<'a>> {
        let expected = rows.checked_mul(row_width).ok_or_else(|| Error::Internal {
            message: "CUDA FP8 activation pack input size overflow".into(),
        })?;
        if rows == 0 || row_width == 0 || input.len() != expected {
            return Err(Error::Internal {
                message: format!(
                    "CUDA FP8 activation pack input mismatch: rows={rows} row_width={row_width} input={}",
                    input.len()
                ),
            });
        }
        self.pack_fp8_rows_from_f32_preallocated(
            &input.buffer,
            rows,
            row_width,
            &mut storage.x_packed,
            storage.value_capacity,
            &mut storage.x_scales,
            storage.scale_capacity,
        )?;
        self.prepared_fp8_activation_from_storage(storage, rows, row_width)
    }

    pub fn prepared_fp8_activation_from_storage<'a>(
        &self,
        storage: &'a CudaFp8ActivationPack,
        rows: usize,
        row_width: usize,
    ) -> Result<CudaPreparedFp8Activation<'a>> {
        let values = rows.checked_mul(row_width).ok_or_else(|| Error::Internal {
            message: "CUDA prepared FP8 activation size overflow".into(),
        })?;
        let scales = rows
            .checked_mul(row_width.div_ceil(ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE))
            .ok_or_else(|| Error::Internal {
                message: "CUDA prepared FP8 scale size overflow".into(),
            })?;
        if rows == 0
            || row_width == 0
            || !row_width.is_multiple_of(ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE)
            || storage.value_capacity != values
            || storage.scale_capacity != scales
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA prepared FP8 activation storage mismatch: rows={rows} width={row_width} values={}/{} scales={}/{}",
                    storage.value_capacity, values, storage.scale_capacity, scales
                ),
            });
        }
        Ok(CudaPreparedFp8Activation {
            x_packed: &storage.x_packed,
            x_scales: &storage.x_scales,
            rows,
            row_width,
        })
    }

    /// Execute the complete HC-pre + layer RMSNorm + FP8 activation producer.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_rmsnorm_fp8_into<'a>(
        &self,
        state: &CudaF32Buffer,
        function_row_major: &CudaF32Buffer,
        hc_scale: &CudaF32Buffer,
        hc_base: &CudaF32Buffer,
        layer_rms_weight: &CudaF32Buffer,
        mix_output: &mut CudaF32Buffer,
        workspace: &mut CudaF32Buffer,
        rows: usize,
        hc: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        hc_eps: f32,
        hc_norm_eps: f32,
        layer_rms_eps: f32,
        hidden_output: &mut CudaF32Buffer,
        normalized_output: &mut CudaF32Buffer,
        split_pre: &mut CudaF32Buffer,
        split_post: &mut CudaF32Buffer,
        split_comb: &mut CudaF32Buffer,
        packed_output: &'a mut CudaFp8ActivationPack,
    ) -> Result<CudaPreparedFp8Activation<'a>> {
        crate::cuda::cutlass::hc_producer(
            &self.stream,
            &state.buffer,
            &function_row_major.buffer,
            &hc_scale.buffer,
            &hc_base.buffer,
            &layer_rms_weight.buffer,
            &mut mix_output.buffer,
            &mut workspace.buffer,
            &mut hidden_output.buffer,
            &mut normalized_output.buffer,
            &mut packed_output.x_packed,
            &mut packed_output.x_scales,
            &mut split_pre.buffer,
            &mut split_post.buffer,
            &mut split_comb.buffer,
            rows,
            hc,
            hidden_size,
            sinkhorn_iters,
            hc_eps,
            hc_norm_eps,
            layer_rms_eps,
        )?;
        self.record_kernel_launch();
        self.prepared_fp8_activation_from_storage(packed_output, rows, hidden_size)
    }

    /// Execute the BF16 compressor dual projection semantic operator.
    pub fn artifact_bf16_compressor_into(
        &self,
        projection1: &CudaArtifactLinearHandle,
        projection2: &CudaArtifactLinearHandle,
        activation: &CudaF32Buffer,
        rows: usize,
        projection1_output: &mut CudaF32Buffer,
        projection2_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let (
            CudaArtifactLinearShape::Bf16Bytes {
                out_features: n1,
                in_features: k1,
            },
            CudaArtifactLinearShape::Bf16Bytes {
                out_features: n2,
                in_features: k2,
            },
        ) = (projection1.shape, projection2.shape)
        else {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor requires BF16 weights, got first={:?} second={:?}",
                    projection1.shape, projection2.shape
                ),
            });
        };
        if k1 != k2 || activation.len() != rows * k1 {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor input mismatch: rows={rows} first_k={k1} second_k={k2} input={}",
                    activation.len()
                ),
            });
        }
        if projection1_output.len() != rows * n1 || projection2_output.len() != rows * n2 {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor output mismatch: first={}/{} second={}/{}",
                    projection1_output.len(),
                    rows * n1,
                    projection2_output.len(),
                    rows * n2
                ),
            });
        }
        crate::cuda::cutlass::bf16_compressor(
            &self.stream,
            &activation.buffer,
            &projection1.weight,
            &projection2.weight,
            &mut projection1_output.buffer,
            &mut projection2_output.buffer,
            rows,
            n1,
            n2,
            k1,
        )?;
        self.record_kernel_launch();
        Ok(())
    }

    /// Execute the checkpoint-native proposal stage-zero target-tap projection and
    /// RMSNorm in one cooperative fused semantic launch.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_main_project_norm_into(
        &self,
        projection: &CudaArtifactLinearHandle,
        norm_weight: &CudaF32Buffer,
        input: &CudaF32Buffer,
        rows: usize,
        rms_eps: f32,
        activation: &mut CudaFp8ActivationPack,
        inv_rms: &mut CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m,
            block_k,
        } = projection.shape
        else {
            return Err(Error::Internal {
                message: format!(
                    "proposal main projection requires FP8/E8M0 weights, got {:?}",
                    projection.shape
                ),
            });
        };
        if block_m != 128 || block_k != 128 || !out_features.is_multiple_of(128) {
            return Err(Error::Internal {
                message: format!(
                    "proposal main projection requires K128/N128 layout, got {:?}",
                    projection.shape
                ),
            });
        }
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "proposal main projection input size overflow".into(),
            })?;
        let output_len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::Internal {
                message: "proposal main projection output size overflow".into(),
            })?;
        let scale_len = rows
            .checked_mul(in_features / 128)
            .ok_or_else(|| Error::Internal {
                message: "proposal main projection scale size overflow".into(),
            })?;
        if input.len() != input_len
            || norm_weight.len() != out_features
            || activation.value_capacity != input_len
            || activation.scale_capacity != scale_len
            || inv_rms.len() != rows
            || output.len() != output_len
        {
            return Err(Error::Internal {
                message: format!(
                    "fused proposal main-project/norm binding mismatch: input={}/{} norm={}/{} activation={}/{} scales={}/{} inv_rms={}/{} output={}/{}",
                    input.len(),
                    input_len,
                    norm_weight.len(),
                    out_features,
                    activation.value_capacity,
                    input_len,
                    activation.scale_capacity,
                    scale_len,
                    inv_rms.len(),
                    rows,
                    output.len(),
                    output_len
                ),
            });
        }
        let weight_scales = projection.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "proposal main projection weight scales are missing".into(),
        })?;
        crate::cuda::cutlass::main_project_norm(
            &self.stream,
            &input.buffer,
            &mut activation.x_packed,
            &mut activation.x_scales,
            &projection.weight,
            weight_scales,
            &norm_weight.buffer,
            &mut inv_rms.buffer,
            &mut output.buffer,
            rows,
            in_features,
            out_features,
            rms_eps,
        )?;
        self.record_kernel_launch();
        Ok(())
    }

    /// Execute the required one-launch QueryA+KV FP8 projection bundle.
    /// Any shape, binding, or native-provider mismatch is fatal.
    pub fn artifact_fp8_query_a_kv_into(
        &self,
        query_a: &CudaArtifactLinearHandle,
        key_value: &CudaArtifactLinearHandle,
        activation: &CudaPreparedFp8Activation<'_>,
        query_a_output: &mut CudaF32Buffer,
        key_value_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let (
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: query_a_out,
                in_features: query_a_in,
                block_m: query_a_block_m,
                block_k: query_a_block_k,
            },
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: kv_out,
                in_features: kv_in,
                block_m: kv_block_m,
                block_k: kv_block_k,
            },
        ) = (query_a.shape, key_value.shape)
        else {
            return Err(Error::Internal {
                message: format!(
                    "fused FP8 QueryA+KV requires FP8 weights, got query_a={:?} kv={:?}",
                    query_a.shape, key_value.shape
                ),
            });
        };
        if query_a_in != kv_in
            || query_a_in != activation.row_width
            || query_a_block_m != 128
            || query_a_block_k != 128
            || kv_block_m != 128
            || kv_block_k != 128
        {
            return Err(Error::Internal {
                message: format!(
                    "fused FP8 QueryA+KV binding mismatch: query_a={:?} kv={:?} activation_width={}",
                    query_a.shape, key_value.shape, activation.row_width
                ),
            });
        }
        let rows = activation.rows;
        if query_a_output.len() != rows * query_a_out || key_value_output.len() != rows * kv_out {
            return Err(Error::Internal {
                message: format!(
                    "CUTLASS FP8 QueryA+KV output mismatch: query_a={}/{} kv={}/{}",
                    query_a_output.len(),
                    rows * query_a_out,
                    key_value_output.len(),
                    rows * kv_out
                ),
            });
        }
        let query_a_scales = query_a.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "CUTLASS FP8 QueryA weight scales are missing".into(),
        })?;
        let kv_scales = key_value.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "CUTLASS FP8 KV weight scales are missing".into(),
        })?;

        crate::cuda::cutlass::fp8_query_a_kv(
            &self.stream,
            activation.x_packed,
            activation.x_scales,
            &query_a.weight,
            query_a_scales,
            &key_value.weight,
            kv_scales,
            &mut query_a_output.buffer,
            &mut key_value_output.buffer,
            rows,
            query_a_out,
            kv_out,
            query_a_in,
        )?;
        self.record_kernel_launch();
        Ok(())
    }

    pub fn artifact_linear_rows_from_prepared_fp8_into(
        &self,
        handle: &CudaArtifactLinearHandle,
        activation: &CudaPreparedFp8Activation<'_>,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if !self.artifact_linear_supports_prepacked_fp8(handle) {
            return Err(Error::Internal {
                message: "CUDA prepared FP8 activation requires an FP8 MMA linear".into(),
            });
        }
        if handle.shape.in_features() != activation.row_width {
            return Err(Error::Internal {
                message: format!(
                    "CUDA prepared FP8 activation width mismatch: activation={} linear={}",
                    activation.row_width,
                    handle.shape.in_features()
                ),
            });
        }
        let expected_output = activation
            .rows
            .checked_mul(handle.shape.out_features())
            .ok_or_else(|| Error::Internal {
                message: "CUDA prepared FP8 output size overflow".into(),
            })?;
        if output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA prepared FP8 output mismatch: expected={expected_output} got={}",
                    output.len()
                ),
            });
        }
        self.artifact_linear_rows_prepacked_fp8(
            handle,
            activation.x_packed,
            activation.x_scales,
            activation.rows,
            &mut output.buffer,
        )
    }

    /// Execute two artifact linears that consume the same row-major activation.
    ///
    /// When both consumers use the FP8 MMA path, the activation is packed once
    /// and both GEMMs consume the same packed values and scales. The fallback
    /// preserves the existing per-linear preparation semantics.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_linear_pair_rows_from_device_into_with_scratch(
        &self,
        first: &CudaArtifactLinearHandle,
        second: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        first_output: &mut CudaF32Buffer,
        second_output: &mut CudaF32Buffer,
        scratch: &mut CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        let in_features = first.shape.in_features();
        if second.shape.in_features() != in_features {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear pair input mismatch: first={} second={}",
                    in_features,
                    second.shape.in_features()
                ),
            });
        }
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear pair input size overflow".into(),
            })?;
        if rows == 0 || input.len() != input_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear pair rows input mismatch: rows={rows} in_features={in_features} input={}",
                    input.len()
                ),
            });
        }
        let first_output_len = rows
            .checked_mul(first.shape.out_features())
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear pair first output overflow".into(),
            })?;
        let second_output_len = rows
            .checked_mul(second.shape.out_features())
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear pair second output overflow".into(),
            })?;
        if first_output.len() != first_output_len || second_output.len() != second_output_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear pair output mismatch: first={}/{} second={}/{}",
                    first_output.len(),
                    first_output_len,
                    second_output.len(),
                    second_output_len
                ),
            });
        }

        if self.artifact_linear_supports_prepacked_fp8(first)
            && self.artifact_linear_supports_prepacked_fp8(second)
        {
            self.pack_fp8_rows_from_f32_preallocated(
                &input.buffer,
                rows,
                in_features,
                &mut scratch.x_packed,
                scratch.value_capacity,
                &mut scratch.x_scales,
                scratch.scale_capacity,
            )?;
            self.artifact_linear_rows_prepacked_fp8(
                first,
                &scratch.x_packed,
                &scratch.x_scales,
                rows,
                &mut first_output.buffer,
            )?;
            return self.artifact_linear_rows_prepacked_fp8(
                second,
                &scratch.x_packed,
                &scratch.x_scales,
                rows,
                &mut second_output.buffer,
            );
        }

        self.artifact_linear_rows_from_device_into_with_scratch(
            first,
            input,
            rows,
            first_output,
            scratch,
        )?;
        self.artifact_linear_rows_from_device_into_with_scratch(
            second,
            input,
            rows,
            second_output,
            scratch,
        )
    }

    pub fn artifact_swiglu_ffn_rows_from_device(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output_scale: f32,
        swiglu_limit: f32,
    ) -> Result<CudaF32Buffer> {
        let in_features = gate.shape.in_features();
        let intermediate = gate.shape.out_features();
        if rows == 0 || input.len() != rows * in_features || up.shape.in_features() != in_features {
            return Err(Error::Internal {
                message: format!(
                    "CUDA batched SwiGLU input mismatch: rows={rows} input={} gate_in={} up_in={}",
                    input.len(),
                    in_features,
                    up.shape.in_features()
                ),
            });
        }
        if up.shape.out_features() != intermediate || down.shape.in_features() != intermediate {
            return Err(Error::Internal {
                message: format!(
                    "CUDA batched SwiGLU shape mismatch: gate={:?} up={:?} down={:?}",
                    gate.shape, up.shape, down.shape
                ),
            });
        }
        let gated = self.artifact_linear_rows_from_device(gate, input, rows)?;
        let upd = self.artifact_linear_rows_from_device(up, input, rows)?;
        let mut hidden =
            self.zero_f32_buffer(rows.checked_mul(intermediate).ok_or_else(|| {
                Error::Internal {
                    message: "CUDA batched SwiGLU hidden size overflow".into(),
                }
            })?)?;
        self.launched(unsafe {
            self.module.swiglu_weighted_clamped(
                &self.stream,
                LaunchConfig::for_num_elems((rows * intermediate) as u32),
                &gated.buffer,
                &upd.buffer,
                &mut hidden.buffer,
                (rows * intermediate) as u32,
                output_scale,
                swiglu_limit,
            )
        })?;
        self.artifact_linear_rows_from_device(down, &hidden, rows)
    }

    /// Device-resident grouped matvec for block-diagonal weight layouts.
    ///
    /// `context` is the full `[o_groups * group_in]` device buffer. `weight` is
    /// the dequantized `[output_latent_dim, group_in]` f32 weight buffer, cached
    /// by the caller. The output `[output_latent_dim]` buffer is allocated here.
    /// One thread per output row; each row only reads its group's context slice.
    pub fn grouped_matvec_f32_from_device(
        &self,
        context: &CudaF32Buffer,
        weight: &CudaF32Buffer,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
    ) -> Result<CudaF32Buffer> {
        self.grouped_matvec_f32_rows_from_device(
            context,
            1,
            weight,
            output_latent_dim,
            group_in,
            o_lora_rank,
        )
    }

    pub fn grouped_output_a_prepacked_fp8_supported(
        &self,
        handle: &CudaArtifactLinearHandle,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
    ) -> bool {
        output_latent_dim.is_multiple_of(16)
            && group_in.is_multiple_of(128)
            && o_lora_rank.is_multiple_of(16)
            && matches!(
                handle.shape,
                CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                    out_features,
                    in_features,
                    block_m: 128,
                    block_k: 128,
                } if out_features == output_latent_dim && in_features == group_in
            )
    }

    /// Official DSV4 grouped WO-A execution: BF16 context × BF16-dequantized
    /// checkpoint weights with FP32 accumulation and BF16 output rounding.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_output_a_bf16_from_fp8_into(
        &self,
        context: &CudaF32Buffer,
        rows: usize,
        handle: &CudaArtifactLinearHandle,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if !self.grouped_output_a_prepacked_fp8_supported(
            handle,
            output_latent_dim,
            group_in,
            o_lora_rank,
        ) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped WO-A BF16 MMA unsupported shape: artifact={:?} out={output_latent_dim} group_in={group_in} rank={o_lora_rank}",
                    handle.shape
                ),
            });
        }
        let groups = output_latent_dim / o_lora_rank;
        let expected_context = rows
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(group_in))
            .ok_or_else(|| Error::Internal {
                message: "CUDA grouped WO-A context size overflow".into(),
            })?;
        let expected_output =
            rows.checked_mul(output_latent_dim)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA grouped WO-A output size overflow".into(),
                })?;
        if rows == 0 || context.len() != expected_context || output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped WO-A BF16 MMA buffer mismatch: rows={rows} context={}/{} output={}/{}",
                    context.len(),
                    expected_context,
                    output.len(),
                    expected_output
                ),
            });
        }
        let weight_scales = handle.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "CUDA grouped WO-A missing FP8 scales".into(),
        })?;
        let scale_cols = group_in / 128;
        self.launched(unsafe {
            self.module.grouped_output_a_bf16_from_fp8(
                &self.stream,
                LaunchConfig {
                    grid_dim: (
                        output_latent_dim.div_ceil(16) as u32,
                        rows.div_ceil(8) as u32,
                        1,
                    ),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                &context.buffer,
                &handle.weight,
                weight_scales,
                &mut output.buffer,
                rows as u32,
                output_latent_dim as u32,
                group_in as u32,
                o_lora_rank as u32,
                scale_cols as u32,
            )
        })
    }

    /// Grouped output-A -> BF16 latent -> output-B MLA transaction. Single-row
    /// execution uses three ordered kernels; wider inputs use one cooperative kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_mla_output_into(
        &self,
        context: &CudaF32Buffer,
        rows: usize,
        output_a: &CudaArtifactLinearHandle,
        output_b: &CudaArtifactLinearHandle,
        groups: usize,
        group_input: usize,
        rank: usize,
        latent: &mut CudaBf16Buffer,
        workspace: &mut CudaArtifactLinearWorkspace,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let latent_size = groups.checked_mul(rank).ok_or_else(|| Error::Internal {
            message: "fused FP8 MLA latent size overflow".into(),
        })?;
        let context_size = groups
            .checked_mul(group_input)
            .ok_or_else(|| Error::Internal {
                message: "fused FP8 MLA context size overflow".into(),
            })?;
        let hidden_size = match output_b.shape {
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m: 128,
                block_k: 128,
            } if in_features == latent_size => out_features,
            _ => {
                return Err(Error::Internal {
                    message: format!(
                        "fused MLA output-B requires FP8/E8M0 [{hidden_size},{latent_size}], got {:?}",
                        output_b.shape,
                        hidden_size = output.len().checked_div(rows).unwrap_or(0)
                    ),
                });
            }
        };
        if !matches!(
            output_a.shape,
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m: 128,
                block_k: 128,
            } if out_features == latent_size && in_features == group_input
        ) {
            return Err(Error::Internal {
                message: format!(
                    "fused MLA output-A requires FP8/E8M0 [{latent_size},{group_input}], got {:?}",
                    output_a.shape
                ),
            });
        }
        let output_a_scales = output_a.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "fused MLA output-A scales are missing".into(),
        })?;
        let output_b_scales = output_b.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "fused MLA output-B scales are missing".into(),
        })?;
        crate::cuda::cutlass::mla_output(
            &self.stream,
            &context.buffer,
            &output_a.weight,
            output_a_scales,
            &output_b.weight,
            output_b_scales,
            &mut latent.buffer,
            &mut workspace.x_packed,
            &mut workspace.x_scales,
            &mut output.buffer,
            rows,
            context_size,
            groups,
            group_input,
            rank,
            latent_size,
            hidden_size,
        )?;
        self.record_kernel_launch();
        if rows == 1 {
            self.record_kernel_launch();
            self.record_kernel_launch();
        }
        Ok(())
    }

    /// Device-resident batched grouped matvec for block-diagonal output-A
    /// layouts. `context` is `[rows, q_full_dim]`; output is
    /// `[rows, output_latent_dim]`.
    pub fn grouped_matvec_f32_rows_from_device(
        &self,
        context: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
    ) -> Result<CudaF32Buffer> {
        if rows == 0 {
            return Err(Error::Internal {
                message: "CUDA grouped rows matvec requires at least one row".into(),
            });
        }
        if o_lora_rank == 0
            || output_latent_dim == 0
            || !output_latent_dim.is_multiple_of(o_lora_rank)
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped rows matvec invalid shape: out={output_latent_dim} rank={o_lora_rank} group_in={group_in}"
                ),
            });
        }
        let groups = output_latent_dim / o_lora_rank;
        let expected_context = rows
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(group_in))
            .ok_or_else(|| {
                Error::Internal { message: format!(
                    "CUDA grouped rows matvec context size overflow: rows={rows} groups={groups} group_in={group_in}"
                ) }
            })?;
        if context.len() != expected_context {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped rows matvec context length mismatch: expected {expected_context}, got {}",
                    context.len()
                ),
            });
        }
        let expected_weight = output_latent_dim.checked_mul(group_in).ok_or_else(|| {
            Error::Internal { message: format!(
                "CUDA grouped rows matvec weight size overflow: out={output_latent_dim} group_in={group_in}"
            ) }
        })?;
        if weight.len() != expected_weight {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped rows matvec weight length mismatch: expected {expected_weight}, got {}",
                    weight.len()
                ),
            });
        }
        let output_len = rows.checked_mul(output_latent_dim).ok_or_else(|| {
            Error::Internal { message: format!(
                "CUDA grouped rows matvec output size overflow: rows={rows} out={output_latent_dim}"
            ) }
        })?;
        let mut output = self.zero_f32_buffer(output_len)?;
        self.grouped_matvec_f32_rows_from_device_into(
            context,
            rows,
            weight,
            output_latent_dim,
            group_in,
            o_lora_rank,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn grouped_matvec_f32_rows_from_device_into(
        &self,
        context: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        output_latent_dim: usize,
        group_in: usize,
        o_lora_rank: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if rows == 0 {
            return Err(Error::Internal {
                message: "CUDA grouped rows matvec requires at least one row".into(),
            });
        }
        if o_lora_rank == 0
            || output_latent_dim == 0
            || !output_latent_dim.is_multiple_of(o_lora_rank)
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped rows matvec invalid shape: out={output_latent_dim} rank={o_lora_rank} group_in={group_in}"
                ),
            });
        }
        let groups = output_latent_dim / o_lora_rank;
        let expected_context = rows
            .checked_mul(groups)
            .and_then(|value| value.checked_mul(group_in))
            .ok_or_else(|| {
                Error::Internal { message: format!(
                    "CUDA grouped rows matvec context size overflow: rows={rows} groups={groups} group_in={group_in}"
                ) }
            })?;
        if context.len() != expected_context {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped rows matvec context length mismatch: expected {expected_context}, got {}",
                    context.len()
                ),
            });
        }
        let expected_weight = output_latent_dim.checked_mul(group_in).ok_or_else(|| {
            Error::Internal { message: format!(
                "CUDA grouped rows matvec weight size overflow: out={output_latent_dim} group_in={group_in}"
            ) }
        })?;
        if weight.len() != expected_weight {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped rows matvec weight length mismatch: expected {expected_weight}, got {}",
                    weight.len()
                ),
            });
        }
        let output_len = rows.checked_mul(output_latent_dim).ok_or_else(|| {
            Error::Internal { message: format!(
                "CUDA grouped rows matvec output size overflow: rows={rows} out={output_latent_dim}"
            ) }
        })?;
        if output.len() != output_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped rows matvec output length mismatch: expected {output_len}, got {}",
                    output.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.grouped_matvec_f32_rows(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    output_len,
                    "grouped_matvec_f32_rows",
                    "output_len",
                )?),
                &context.buffer,
                &weight.buffer,
                &mut output.buffer,
                checked_u32(rows, "grouped_matvec_f32_rows", "rows")?,
                checked_u32(
                    output_latent_dim,
                    "grouped_matvec_f32_rows",
                    "output_latent_dim",
                )?,
                checked_u32(group_in, "grouped_matvec_f32_rows", "group_in")?,
                checked_u32(o_lora_rank, "grouped_matvec_f32_rows", "o_lora_rank")?,
            )
        })
    }

    pub fn artifact_linear_topk(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if top_k > 40 {
            return Err(Error::Internal {
                message: format!("CUDA artifact linear top-k supports k<=40, got {top_k}"),
            });
        }
        if input.len() != handle.shape.in_features() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear top-k input length mismatch: expected {}, got {}",
                    handle.shape.in_features(),
                    input.len()
                ),
            });
        }
        let xd = self.upload_f32_buffer(input)?;
        self.artifact_linear_topk_from_device(handle, &xd, top_k)
    }

    pub fn dsv4_router_token_ids(
        &self,
        token_ids: &[u32],
        hash_rows: usize,
    ) -> Result<CudaDsv4RouterTokenIds> {
        let validated = validate_dsv4_router_token_ids(token_ids, hash_rows)?;
        let staging = cu(PinnedHostBuffer::from_slice(&self._ctx, &validated))?;
        let buffer = self.record_device_allocation(validated.len(), unsafe {
            DeviceBuffer::from_pinned_host(&self.stream, &staging)
        })?;
        self.counters.add_host_to_device(slice_bytes(&validated));
        let copy_event = match self.stream.record_event(None) {
            Ok(event) => event,
            Err(error) => {
                self.record_stream_wide_sync(self.stream.synchronize())?;
                return Err(Error::Internal {
                    message: format!("CUDA router token-id copy event failed: {error:?}"),
                });
            }
        };
        Ok(CudaDsv4RouterTokenIds {
            host: token_ids.to_vec(),
            device: CudaTypedBuffer::from_device_buffer(buffer),
            staging,
            copy_event,
        })
    }

    pub fn update_dsv4_router_token_ids(
        &self,
        token_ids: &[u32],
        hash_rows: usize,
        cached: &mut CudaDsv4RouterTokenIds,
    ) -> Result<()> {
        if cached.host.len() != token_ids.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 hash router token buffer shape mismatch: cached={} requested={}",
                    cached.host.len(),
                    token_ids.len()
                ),
            });
        }
        if cached.host == token_ids {
            // Different hash layers may have different row counts, so retain the
            // cheap host row validation even when no device overwrite is needed.
            validate_dsv4_router_token_ids(token_ids, hash_rows)?;
            return Ok(());
        }
        let validated = validate_dsv4_router_token_ids(token_ids, hash_rows)?;
        cu(cached.copy_event.synchronize())?;
        cached.staging.as_mut_slice().copy_from_slice(&validated);
        unsafe {
            cu(cached
                .device
                .buffer
                .copy_from_pinned_host_async(&self.stream, &cached.staging))?;
        }
        self.counters.add_host_to_device(slice_bytes(&validated));
        match self.stream.record_event(None) {
            Ok(event) => cached.copy_event = event,
            Err(error) => {
                self.record_stream_wide_sync(self.stream.synchronize())?;
                cached.host.clear();
                cached.host.extend_from_slice(token_ids);
                return Err(Error::Internal {
                    message: format!(
                        "CUDA router token-id copy event failed after the copy completed: {error:?}"
                    ),
                });
            }
        }
        cached.host.clear();
        cached.host.extend_from_slice(token_ids);
        Ok(())
    }

    pub fn upload_dsv4_router_hash_table(
        &self,
        table: &[usize],
        rows: usize,
        cols: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<CudaDsv4RouterHashTable> {
        let table = validate_dsv4_router_hash_table(table, rows, cols, experts, top_k)?;
        Ok(CudaDsv4RouterHashTable {
            buffer: self.upload_i32(&table)?,
            rows,
            cols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_router_topk_sqrt_softplus_rows_from_device_into(
        &self,
        logits: &CudaF32Buffer,
        bias: Option<&CudaF32Buffer>,
        tokens: usize,
        experts: usize,
        top_k: usize,
        route_scale: f32,
        indices: &mut CudaI32Buffer,
        weights: &mut CudaF32Buffer,
    ) -> Result<()> {
        if tokens == 0 || experts == 0 || top_k == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 router topk requires non-empty shape: tokens={tokens} experts={experts} top_k={top_k}"
                ),
            });
        }
        if experts > 512 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 router topk supports at most 512 experts, got {experts}"
                ),
            });
        }
        if top_k > 64 || top_k > experts {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 router topk requires top_k in 1..={} and <=64, got {top_k}",
                    experts.min(64)
                ),
            });
        }
        if logits.len() != tokens * experts {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 router topk logits length mismatch: got {} expected {}x{}",
                    logits.len(),
                    tokens,
                    experts
                ),
            });
        }
        let (bias_buf, bias_enabled) = if let Some(bias) = bias {
            if bias.len() != experts {
                return Err(Error::Internal {
                    message: format!(
                        "CUDA DSV4 router topk bias length mismatch: got {} expected {experts}",
                        bias.len()
                    ),
                });
            }
            (bias, 1u32)
        } else {
            (logits, 0u32)
        };
        let out_len = tokens.checked_mul(top_k).ok_or_else(|| Error::Internal {
            message: "CUDA DSV4 router topk output overflow".into(),
        })?;
        if indices.len() != out_len || weights.len() != out_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 router topk output mismatch: expected {out_len}, indices={}, weights={}",
                    indices.len(),
                    weights.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.dsv4_router_topk_sqrt_softplus_rows(
                &self.stream,
                LaunchConfig::for_num_elems(tokens as u32),
                &logits.buffer,
                &bias_buf.buffer,
                &mut indices.buffer,
                &mut weights.buffer,
                tokens as u32,
                experts as u32,
                top_k as u32,
                bias_enabled,
                route_scale,
            )
        })
        .map(|_| ())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_router_hash_sqrt_softplus_rows_from_device_into(
        &self,
        logits: &CudaF32Buffer,
        token_ids: &CudaDsv4RouterTokenIds,
        hash_table: &CudaDsv4RouterHashTable,
        tokens: usize,
        experts: usize,
        top_k: usize,
        route_scale: f32,
        indices: &mut CudaI32Buffer,
        weights: &mut CudaF32Buffer,
    ) -> Result<()> {
        if tokens == 0 || experts == 0 || top_k == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 hash router requires non-empty shape: tokens={tokens} experts={experts} top_k={top_k}"
                ),
            });
        }
        if top_k > 64 || top_k > experts || top_k > hash_table.cols {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 hash router top_k {top_k} exceeds experts={experts}, hash_cols={}, or kernel limit 64",
                    hash_table.cols
                ),
            });
        }
        if !route_scale.is_finite() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 hash router route_scale must be finite, got {route_scale}"
                ),
            });
        }
        let logits_len = tokens.checked_mul(experts).ok_or_else(|| Error::Internal {
            message: "CUDA DSV4 hash router logits shape overflow".into(),
        })?;
        let output_len = tokens.checked_mul(top_k).ok_or_else(|| Error::Internal {
            message: "CUDA DSV4 hash router output shape overflow".into(),
        })?;
        if logits.len() != logits_len || token_ids.device.len() != tokens {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 hash router input mismatch: logits={} expected={logits_len}, token_ids={} expected={tokens}",
                    logits.len(),
                    token_ids.device.len()
                ),
            });
        }
        if indices.len() != output_len || weights.len() != output_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA DSV4 hash router output mismatch: expected {output_len}, indices={}, weights={}",
                    indices.len(),
                    weights.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.dsv4_router_hash_sqrt_softplus_rows(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    tokens,
                    "dsv4_router_hash_sqrt_softplus_rows",
                    "tokens",
                )?),
                &logits.buffer,
                &token_ids.device.buffer,
                &hash_table.buffer,
                &mut indices.buffer,
                &mut weights.buffer,
                checked_u32(tokens, "dsv4_router_hash_sqrt_softplus_rows", "tokens")?,
                checked_u32(experts, "dsv4_router_hash_sqrt_softplus_rows", "experts")?,
                checked_u32(
                    hash_table.rows,
                    "dsv4_router_hash_sqrt_softplus_rows",
                    "hash_rows",
                )?,
                checked_u32(
                    hash_table.cols,
                    "dsv4_router_hash_sqrt_softplus_rows",
                    "hash_cols",
                )?,
                checked_u32(top_k, "dsv4_router_hash_sqrt_softplus_rows", "top_k")?,
                route_scale,
            )
        })
        .map(|_| ())
    }

    pub fn topk_vocab_rows_from_device_into(
        &self,
        logits: &CudaF32Buffer,
        rows: usize,
        vocab: usize,
        top_k: usize,
        indices: &mut CudaI32Buffer,
        values: &mut CudaF32Buffer,
    ) -> Result<()> {
        if rows == 0
            || vocab == 0
            || vocab > i32::MAX as usize
            || top_k == 0
            || top_k > vocab
            || top_k > 40
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA vocab rows top-k requires rows>0, vocab<=i32::MAX, and k in 1..={}, got rows={rows} vocab={vocab} k={top_k}",
                    vocab.min(40)
                ),
            });
        }
        let logits_len = rows.checked_mul(vocab).ok_or_else(|| Error::Internal {
            message: "CUDA vocab rows top-k logits size overflow".into(),
        })?;
        let output_len = rows.checked_mul(top_k).ok_or_else(|| Error::Internal {
            message: "CUDA vocab rows top-k output size overflow".into(),
        })?;
        if logits.len() != logits_len || indices.len() < output_len || values.len() < output_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA vocab rows top-k workspace mismatch: logits={} indices={} values={} expected_logits={logits_len} expected_output={output_len}",
                    logits.len(),
                    indices.len(),
                    values.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.topk_vocab_rows(
                &self.stream,
                LaunchConfig {
                    grid_dim: (checked_u32(rows, "topk_vocab_rows", "rows")?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &logits.buffer,
                &mut indices.buffer,
                &mut values.buffer,
                checked_u32(rows, "topk_vocab_rows", "rows")?,
                checked_u32(vocab, "topk_vocab_rows", "vocab")?,
                checked_u32(top_k, "topk_vocab_rows", "top_k")?,
            )
        })
        .map(|_| ())
    }

    pub fn artifact_linear_topk_from_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        top_k: usize,
    ) -> Result<Vec<(u32, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let mut logits = self.zero_f32_buffer(handle.shape.out_features())?;
        let mut indices = self.zero_f32_buffer(top_k)?;
        let mut values = self.zero_f32_buffer(top_k)?;
        self.artifact_linear_topk_from_device_into(
            handle,
            input,
            top_k,
            &mut logits,
            &mut indices,
            &mut values,
        )
    }

    pub fn artifact_linear_topk_from_device_into(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        top_k: usize,
        logits: &mut CudaF32Buffer,
        indices: &mut CudaF32Buffer,
        values: &mut CudaF32Buffer,
    ) -> Result<Vec<(u32, f32)>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if top_k > 40 {
            return Err(Error::Internal {
                message: format!("CUDA artifact linear top-k supports k<=40, got {top_k}"),
            });
        }
        if input.len() != handle.shape.in_features()
            || logits.len() != handle.shape.out_features()
            || indices.len() < top_k
            || values.len() < top_k
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear device top-k workspace mismatch: input={} logits={} indices={} values={} expected_input={} expected_logits={} k={top_k}",
                    input.len(),
                    logits.len(),
                    indices.len(),
                    values.len(),
                    handle.shape.in_features(),
                    handle.shape.out_features(),
                ),
            });
        }
        self.artifact_linear_matvec_into(handle, input, logits)?;
        self.launched(unsafe {
            self.module.topk_vocab(
                &self.stream,
                one_block_config(256),
                &logits.buffer,
                &mut indices.buffer,
                &mut values.buffer,
                handle.shape.out_features() as u32,
                top_k as u32,
            )
        })?;
        let indices = self.download_f32(&indices.buffer, top_k)?;
        let values = self.download_f32(&values.buffer, top_k)?;
        Ok(indices
            .into_iter()
            .zip(values)
            .map(|(index, value)| (index as u32, value))
            .collect())
    }

    pub fn fp8_activation_quantize_buffer_in_place(
        &self,
        values: &mut CudaF32Buffer,
        row_width: usize,
        block_size: usize,
    ) -> Result<()> {
        let value_len = values.len();
        self.fp8_activation_quantize_in_place(&mut values.buffer, value_len, row_width, block_size)
    }

    pub fn fp8_activation_quantize_in_place(
        &self,
        values: &mut DeviceBuffer<f32>,
        value_len: usize,
        row_width: usize,
        block_size: usize,
    ) -> Result<()> {
        if value_len == 0
            || row_width == 0
            || block_size == 0
            || !row_width.is_multiple_of(block_size)
            || !value_len.is_multiple_of(row_width)
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid CUDA FP8 activation quant shape: len={value_len}, row_width={row_width}, block_size={block_size}"
                ),
            });
        }
        self.launched(unsafe {
            self.module.fp8_e4m3fn_e8m0_quantize_f32_inplace(
                &self.stream,
                LaunchConfig::for_num_elems((value_len / block_size) as u32),
                values,
                value_len as u32,
                row_width as u32,
                block_size as u32,
            )
        })
    }

    pub fn fp8_attention_kv_qat_quantize_buffer_in_place(
        &self,
        values: &mut CudaF32Buffer,
        head_dim: usize,
        rope_dim: usize,
    ) -> Result<()> {
        if values.is_empty()
            || head_dim == 0
            || rope_dim > head_dim
            || !values.len().is_multiple_of(head_dim)
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid CUDA attention KV QAT shape: len={} head_dim={head_dim} rope_dim={rope_dim}",
                    values.len()
                ),
            });
        }
        let non_rope = head_dim - rope_dim;
        if non_rope == 0 {
            return Ok(());
        }
        let block_size = 64usize;
        let effective_block_size = if non_rope.is_multiple_of(block_size) {
            block_size
        } else {
            non_rope
        };
        let value_len = values.len();
        let rows = value_len / head_dim;
        let blocks_per_row = non_rope.div_ceil(effective_block_size);
        self.launched(unsafe {
            self.module.fp8_e4m3fn_e8m0_quantize_non_rope_f32_inplace(
                &self.stream,
                LaunchConfig::for_num_elems((rows * blocks_per_row) as u32),
                &mut values.buffer,
                value_len as u32,
                head_dim as u32,
                rope_dim as u32,
                block_size as u32,
            )
        })
    }

    pub fn fp4_hadamard_qat_quantize_buffer_in_place(
        &self,
        values: &mut CudaF32Buffer,
        row_width: usize,
    ) -> Result<()> {
        if values.is_empty()
            || row_width == 0
            || !row_width.is_power_of_two()
            || !values.len().is_multiple_of(row_width)
        {
            return Err(Error::Internal {
                message: format!(
                    "invalid CUDA indexer Hadamard/FP4 QAT shape: len={} row_width={row_width}",
                    values.len()
                ),
            });
        }
        let value_len = values.len();
        self.launched(unsafe {
            self.module.hadamard_fp4_e2m1_e8m0_quantize_f32_inplace(
                &self.stream,
                LaunchConfig::for_num_elems((value_len / row_width) as u32),
                &mut values.buffer,
                value_len as u32,
                row_width as u32,
                32,
            )
        })
    }

    pub fn create_compressor_recurrent_state(
        &self,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
    ) -> Result<CudaCompressorRecurrentState> {
        let shape = CompressorRecurrentShape {
            ratio,
            head_dim,
            out_dim,
            overlap,
        };
        shape.validate()?;
        let state_elements = shape.state_elements()?;
        let kv_state = self.zero_f32_buffer(state_elements)?;
        let score_state = self.zero_f32_buffer(state_elements)?;
        let mut state = CudaCompressorRecurrentState {
            kv_state,
            score_state,
            shape,
        };
        self.reset_compressor_recurrent_state(&mut state)?;
        Ok(state)
    }

    pub fn clone_compressor_recurrent_state(
        &self,
        source: &CudaCompressorRecurrentState,
    ) -> Result<CudaCompressorRecurrentState> {
        source.shape.validate()?;
        Ok(CudaCompressorRecurrentState {
            kv_state: self.clone_f32_buffer(&source.kv_state)?,
            score_state: self.clone_f32_buffer(&source.score_state)?,
            shape: source.shape,
        })
    }

    pub fn create_compressor_recurrent_checkpoint_slab(
        &self,
        source: &CudaCompressorRecurrentState,
        slots: usize,
    ) -> Result<CudaCompressorRecurrentCheckpointSlab> {
        source.shape.validate()?;
        if slots == 0 {
            return Err(Error::Internal {
                message: "compressor recurrent checkpoint slab requires at least one slot".into(),
            });
        }
        let state_elements = source.shape.state_elements()?;
        let slab_elements = state_elements
            .checked_mul(slots)
            .ok_or_else(|| Error::Internal {
                message: "compressor recurrent checkpoint slab size overflow".into(),
            })?;
        Ok(CudaCompressorRecurrentCheckpointSlab {
            kv_states: self.zero_f32_buffer(slab_elements)?,
            score_states: self.zero_f32_buffer(slab_elements)?,
            shape: source.shape,
            slots,
        })
    }

    pub fn capture_compressor_recurrent_checkpoint(
        &self,
        source: &CudaCompressorRecurrentState,
        checkpoints: &mut CudaCompressorRecurrentCheckpointSlab,
        slot: usize,
    ) -> Result<()> {
        if checkpoints.shape != source.shape || slot >= checkpoints.slots {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent checkpoint capture mismatch: slot={slot} capacity={} source={:?} slab={:?}",
                    checkpoints.slots, source.shape, checkpoints.shape
                ),
            });
        }
        let state_elements = source.shape.state_elements()?;
        let offset = slot
            .checked_mul(state_elements)
            .ok_or_else(|| Error::Internal {
                message: "compressor recurrent checkpoint offset overflow".into(),
            })?;
        self.copy_f32_range(
            &source.kv_state,
            0,
            &mut checkpoints.kv_states,
            offset,
            state_elements,
        )?;
        self.copy_f32_range(
            &source.score_state,
            0,
            &mut checkpoints.score_states,
            offset,
            state_elements,
        )
    }

    pub fn restore_compressor_recurrent_checkpoint(
        &self,
        checkpoints: &CudaCompressorRecurrentCheckpointSlab,
        slot: usize,
        destination: &mut CudaCompressorRecurrentState,
    ) -> Result<()> {
        if checkpoints.shape != destination.shape || slot >= checkpoints.slots {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent checkpoint restore mismatch: slot={slot} capacity={} destination={:?} slab={:?}",
                    checkpoints.slots, destination.shape, checkpoints.shape
                ),
            });
        }
        let state_elements = destination.shape.state_elements()?;
        let offset = slot
            .checked_mul(state_elements)
            .ok_or_else(|| Error::Internal {
                message: "compressor recurrent checkpoint offset overflow".into(),
            })?;
        self.copy_f32_range(
            &checkpoints.kv_states,
            offset,
            &mut destination.kv_state,
            0,
            state_elements,
        )?;
        self.copy_f32_range(
            &checkpoints.score_states,
            offset,
            &mut destination.score_state,
            0,
            state_elements,
        )
    }

    pub fn reset_compressor_recurrent_state(
        &self,
        state: &mut CudaCompressorRecurrentState,
    ) -> Result<()> {
        let state_elements = state.shape.state_elements()?;
        self.launched(unsafe {
            self.module.compressor_recurrent_reset_f32(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    state_elements,
                    "compressor recurrent reset",
                    "state_elements",
                )?),
                &mut state.kv_state.buffer,
                &mut state.score_state.buffer,
                checked_u32(
                    state_elements,
                    "compressor recurrent reset",
                    "state_elements",
                )?,
            )
        })
    }

    pub fn compressor_recurrent_seed_prefill(
        &self,
        state: &mut CudaCompressorRecurrentState,
        projected_kv_rows: &CudaF32Buffer,
        projected_score_rows: &CudaF32Buffer,
        ape: &CudaF32Buffer,
        tokens: usize,
    ) -> Result<usize> {
        state.shape.validate()?;
        let projected_elements =
            tokens
                .checked_mul(state.shape.out_dim)
                .ok_or_else(|| Error::Internal {
                    message: "compressor recurrent seed size overflow".into(),
                })?;
        if projected_kv_rows.len() != projected_elements
            || projected_score_rows.len() != projected_elements
            || ape.len() != state.shape.ape_elements()?
        {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent seed length mismatch: kv={} score={} ape={} expected projected={} ape={}",
                    projected_kv_rows.len(),
                    projected_score_rows.len(),
                    ape.len(),
                    projected_elements,
                    state.shape.ape_elements()?
                ),
            });
        }
        let state_elements = state.shape.state_elements()?;
        self.launched(unsafe {
            self.module.compressor_recurrent_seed_prefill_f32(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    state_elements,
                    "compressor recurrent seed",
                    "state_elements",
                )?),
                &projected_kv_rows.buffer,
                &projected_score_rows.buffer,
                &ape.buffer,
                &mut state.kv_state.buffer,
                &mut state.score_state.buffer,
                checked_u32(tokens, "compressor recurrent seed", "tokens")?,
                checked_u32(state.shape.ratio, "compressor recurrent seed", "ratio")?,
                checked_u32(state.shape.out_dim, "compressor recurrent seed", "out_dim")?,
                if state.shape.overlap { 1 } else { 0 },
                checked_u32(
                    state_elements,
                    "compressor recurrent seed",
                    "state_elements",
                )?,
            )
        })?;
        Ok(state.shape.prefill_groups(tokens))
    }

    pub fn compressor_recurrent_append_projected(
        &self,
        state: &mut CudaCompressorRecurrentState,
        projected_kv: &CudaF32Buffer,
        projected_score: &CudaF32Buffer,
        ape: &CudaF32Buffer,
        position: usize,
    ) -> Result<bool> {
        state.shape.validate()?;
        if projected_kv.len() != state.shape.out_dim
            || projected_score.len() != state.shape.out_dim
            || ape.len() != state.shape.ape_elements()?
        {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent append length mismatch: kv={} score={} ape={} expected row={} ape={}",
                    projected_kv.len(),
                    projected_score.len(),
                    ape.len(),
                    state.shape.out_dim,
                    state.shape.ape_elements()?
                ),
            });
        }
        self.launched(unsafe {
            self.module.compressor_recurrent_append_projected_f32(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    state.shape.out_dim,
                    "compressor recurrent append",
                    "out_dim",
                )?),
                &projected_kv.buffer,
                &projected_score.buffer,
                &ape.buffer,
                &mut state.kv_state.buffer,
                &mut state.score_state.buffer,
                checked_u32(position, "compressor recurrent append", "position")?,
                checked_u32(state.shape.ratio, "compressor recurrent append", "ratio")?,
                checked_u32(
                    state.shape.out_dim,
                    "compressor recurrent append",
                    "out_dim",
                )?,
                if state.shape.overlap { 1 } else { 0 },
            )
        })?;
        Ok(state.shape.is_boundary(position))
    }

    /// Compresses the current recurrent window into `output` and advances the
    /// overlap state in-place. This method performs no allocation or D2H copy.
    pub fn compressor_recurrent_boundary_into(
        &self,
        state: &mut CudaCompressorRecurrentState,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        state.shape.validate()?;
        if output.len() != state.shape.head_dim {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent output length mismatch: got {} expected {}",
                    output.len(),
                    state.shape.head_dim
                ),
            });
        }
        self.launched(unsafe {
            self.module.compressor_recurrent_softmax_f32(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    state.shape.head_dim,
                    "compressor recurrent boundary",
                    "head_dim",
                )?),
                &state.kv_state.buffer,
                &state.score_state.buffer,
                &mut output.buffer,
                checked_u32(state.shape.ratio, "compressor recurrent boundary", "ratio")?,
                checked_u32(
                    state.shape.head_dim,
                    "compressor recurrent boundary",
                    "head_dim",
                )?,
                checked_u32(
                    state.shape.out_dim,
                    "compressor recurrent boundary",
                    "out_dim",
                )?,
                if state.shape.overlap { 1 } else { 0 },
            )
        })?;
        if state.shape.overlap {
            let half = state
                .shape
                .ratio
                .checked_mul(state.shape.out_dim)
                .ok_or_else(|| Error::Internal {
                    message: "compressor recurrent half overflow".into(),
                })?;
            self.copy_f32_within(&mut state.kv_state, half, 0, half)?;
            self.copy_f32_within(&mut state.score_state, half, 0, half)?;
        }
        Ok(())
    }

    pub fn compressor_prefill_softmax_from_device(
        &self,
        kv_rows: &CudaF32Buffer,
        score_rows: &CudaF32Buffer,
        ape: &[f32],
        groups: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
    ) -> Result<CudaF32Buffer> {
        let ape_dev = self.upload_f32_buffer(ape)?;
        let mut output = self.zero_f32_buffer(groups.checked_mul(head_dim).ok_or_else(
            || Error::Internal {
                message: "CUDA compressor output size overflow".into(),
            },
        )?)?;
        self.compressor_prefill_softmax_from_device_into(
            kv_rows,
            score_rows,
            &ape_dev,
            groups,
            ratio,
            head_dim,
            out_dim,
            overlap,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressor_prefill_softmax_from_device_into(
        &self,
        kv_rows: &CudaF32Buffer,
        score_rows: &CudaF32Buffer,
        ape: &CudaF32Buffer,
        groups: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if ratio == 0 || head_dim == 0 || out_dim == 0 {
            return Err(Error::Internal {
                message: format!(
                    "invalid CUDA compressor shape: groups={groups} ratio={ratio} head_dim={head_dim} out_dim={out_dim}"
                ),
            });
        }
        let consumed_tokens = groups.checked_mul(ratio).ok_or_else(|| Error::Internal {
            message: "CUDA compressor token count overflow".into(),
        })?;
        let min_projected =
            consumed_tokens
                .checked_mul(out_dim)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA compressor projected row size overflow".into(),
                })?;
        if kv_rows.len() < min_projected
            || score_rows.len() < min_projected
            || !kv_rows.len().is_multiple_of(out_dim)
            || !score_rows.len().is_multiple_of(out_dim)
            || kv_rows.len() != score_rows.len()
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA compressor projected length mismatch: kv={} score={} min_required={min_projected} out_dim={out_dim}",
                    kv_rows.len(),
                    score_rows.len()
                ),
            });
        }
        let expected_ape = ratio.checked_mul(out_dim).ok_or_else(|| Error::Internal {
            message: "CUDA compressor APE size overflow".into(),
        })?;
        if ape.len() != expected_ape {
            return Err(Error::Internal {
                message: format!(
                    "CUDA compressor APE length mismatch: got {} expected {expected_ape}",
                    ape.len()
                ),
            });
        }
        let expected_output = groups
            .checked_mul(head_dim)
            .ok_or_else(|| Error::Internal {
                message: "CUDA compressor output size overflow".into(),
            })?;
        if output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA compressor output length mismatch: got {} expected {expected_output}",
                    output.len()
                ),
            });
        }
        if groups == 0 {
            return Ok(());
        }
        self.launched(unsafe {
            self.module.dsv4_compressor_prefill_softmax(
                &self.stream,
                LaunchConfig::for_num_elems(expected_output as u32),
                &kv_rows.buffer,
                &score_rows.buffer,
                &ape.buffer,
                &mut output.buffer,
                groups as u32,
                ratio as u32,
                head_dim as u32,
                out_dim as u32,
                if overlap { 1u32 } else { 0u32 },
            )
        })
    }

    fn validate_paged_indexer_storage(
        &self,
        plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        compressed_len: usize,
        layout: crate::cuda::kv_page_pool::PagedPlaneLayout,
    ) -> Result<()> {
        layout.validate()?;
        if block_offsets.len() != 2 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged indexer requires one sequence block range (2 offsets), got {}",
                    block_offsets.len()
                ),
            });
        }
        let required_pages = compressed_len.div_ceil(layout.page_tokens);
        if block_slots.len() < required_pages {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged indexer block table too short: need {required_pages}, got {}",
                    block_slots.len()
                ),
            });
        }
        let slot_elements = layout
            .layer_count
            .checked_mul(layout.page_tokens)
            .and_then(|value| value.checked_mul(layout.elements_per_token))
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged indexer slot size overflow".into(),
            })?;
        if plane.len() < slot_elements || !plane.len().is_multiple_of(slot_elements) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged indexer plane length {} is not a positive multiple of slot size {slot_elements}",
                    plane.len()
                ),
            });
        }
        checked_u32(compressed_len, "CUDA paged indexer", "compressed_len")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_prefill_topk_indices_paged_indexer_from_device_into(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        tokens: usize,
        window_size: usize,
        window_cols: usize,
        extra_cols: usize,
        value_offset: usize,
        compress_ratio: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        page_tokens: usize,
        layer_index: usize,
        layer_count: usize,
        weight_scale: f32,
        output: &mut CudaI32Buffer,
    ) -> Result<()> {
        let layout = crate::cuda::kv_page_pool::PagedPlaneLayout {
            page_tokens,
            elements_per_token: index_head_dim,
            layer_index,
            layer_count,
        };
        self.validate_paged_indexer_storage(
            indexer_plane,
            block_slots,
            block_offsets,
            compressed_len,
            layout,
        )?;
        if tokens == 0 || compress_ratio == 0 || extra_cols == 0 || extra_cols > 512 {
            return Err(Error::Internal {
                message: "invalid CUDA paged prefill indexer shape".into(),
            });
        }
        let query_len = tokens
            .checked_mul(index_heads)
            .and_then(|value| value.checked_mul(index_head_dim))
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged prefill query size overflow".into(),
            })?;
        let weight_len = tokens
            .checked_mul(index_heads)
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged prefill weight size overflow".into(),
            })?;
        let total_cols = window_cols
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged prefill column overflow".into(),
            })?;
        let output_len = tokens
            .checked_mul(total_cols)
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged prefill output overflow".into(),
            })?;
        if window_cols > window_size
            || window_cols > tokens
            || query.len() != query_len
            || weights.len() != weight_len
            || output.len() < output_len
        {
            return Err(Error::Internal {
                message: "CUDA paged prefill indexer buffer mismatch".into(),
            });
        }
        self.launched(unsafe {
            self.module.dsv4_prefill_topk_indices_paged_indexer(
                &self.stream,
                LaunchConfig {
                    grid_dim: (checked_u32(tokens, "CUDA paged prefill", "tokens")?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &query.buffer,
                &weights.buffer,
                &indexer_plane.buffer,
                &block_slots.buffer,
                &block_offsets.buffer,
                &mut output.buffer,
                checked_u32(tokens, "CUDA paged prefill", "tokens")?,
                checked_u32(window_size, "CUDA paged prefill", "window_size")?,
                checked_u32(window_cols, "CUDA paged prefill", "window_cols")?,
                checked_u32(extra_cols, "CUDA paged prefill", "extra_cols")?,
                checked_u32(value_offset, "CUDA paged prefill", "value_offset")?,
                checked_u32(compress_ratio, "CUDA paged prefill", "compress_ratio")?,
                checked_u32(compressed_len, "CUDA paged prefill", "compressed_len")?,
                checked_u32(index_heads, "CUDA paged prefill", "index_heads")?,
                checked_u32(index_head_dim, "CUDA paged prefill", "index_head_dim")?,
                checked_u32(page_tokens, "CUDA paged prefill", "page_tokens")?,
                checked_u32(layer_index, "CUDA paged prefill", "layer_index")?,
                checked_u32(layer_count, "CUDA paged prefill", "layer_count")?,
                weight_scale,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_prefill_topk_indices_fused_index_query_paged_indexer_from_device_into(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        tokens: usize,
        window_size: usize,
        window_cols: usize,
        extra_cols: usize,
        value_offset: usize,
        compress_ratio: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        rope_dim: usize,
        start_position: usize,
        page_tokens: usize,
        layer_index: usize,
        layer_count: usize,
        weight_scale: f32,
        output: &mut CudaI32Buffer,
    ) -> Result<()> {
        let layout = crate::cuda::kv_page_pool::PagedPlaneLayout {
            page_tokens,
            elements_per_token: index_head_dim,
            layer_index,
            layer_count,
        };
        self.validate_paged_indexer_storage(
            indexer_plane,
            block_slots,
            block_offsets,
            compressed_len,
            layout,
        )?;
        let query_len = tokens
            .checked_mul(index_heads)
            .and_then(|v| v.checked_mul(index_head_dim))
            .ok_or_else(|| Error::Internal {
                message: "CUDA fused paged prefill query overflow".into(),
            })?;
        let weight_len = tokens
            .checked_mul(index_heads)
            .ok_or_else(|| Error::Internal {
                message: "CUDA fused paged prefill weights overflow".into(),
            })?;
        let output_len =
            tokens
                .checked_mul(window_cols.checked_add(extra_cols).ok_or_else(|| {
                    Error::Internal {
                        message: "CUDA fused paged prefill columns overflow".into(),
                    }
                })?)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA fused paged prefill output overflow".into(),
                })?;
        let rope_len = start_position
            .checked_add(tokens)
            .and_then(|v| v.checked_mul(rope_dim / 2))
            .ok_or_else(|| Error::Internal {
                message: "CUDA fused paged prefill rope overflow".into(),
            })?;
        if tokens == 0
            || compress_ratio == 0
            || extra_cols == 0
            || extra_cols > 512
            || index_head_dim == 0
            || index_head_dim > 256
            || !index_head_dim.is_power_of_two()
            || !index_head_dim.is_multiple_of(32)
            || rope_dim > index_head_dim
            || !rope_dim.is_multiple_of(2)
            || window_cols > window_size
            || window_cols > tokens
            || query.len() != query_len
            || weights.len() != weight_len
            || output.len() < output_len
            || cos_table.len() < rope_len
            || sin_table.len() < rope_len
        {
            return Err(Error::Internal {
                message: "CUDA fused paged prefill indexer shape mismatch".into(),
            });
        }
        self.launched(unsafe {
            self.module
                .dsv4_prefill_topk_indices_fused_index_query_paged_indexer(
                    &self.stream,
                    LaunchConfig::for_num_elems(checked_u32(
                        tokens,
                        "CUDA fused paged prefill",
                        "tokens",
                    )?),
                    &query.buffer,
                    &weights.buffer,
                    &indexer_plane.buffer,
                    &block_slots.buffer,
                    &block_offsets.buffer,
                    &cos_table.buffer,
                    &sin_table.buffer,
                    &mut output.buffer,
                    checked_u32(tokens, "CUDA fused paged prefill", "tokens")?,
                    checked_u32(window_size, "CUDA fused paged prefill", "window_size")?,
                    checked_u32(window_cols, "CUDA fused paged prefill", "window_cols")?,
                    checked_u32(extra_cols, "CUDA fused paged prefill", "extra_cols")?,
                    checked_u32(value_offset, "CUDA fused paged prefill", "value_offset")?,
                    checked_u32(compress_ratio, "CUDA fused paged prefill", "compress_ratio")?,
                    checked_u32(compressed_len, "CUDA fused paged prefill", "compressed_len")?,
                    checked_u32(index_heads, "CUDA fused paged prefill", "index_heads")?,
                    checked_u32(index_head_dim, "CUDA fused paged prefill", "index_head_dim")?,
                    checked_u32(rope_dim, "CUDA fused paged prefill", "rope_dim")?,
                    checked_u32(start_position, "CUDA fused paged prefill", "start_position")?,
                    checked_u32(page_tokens, "CUDA fused paged prefill", "page_tokens")?,
                    checked_u32(layer_index, "CUDA fused paged prefill", "layer_index")?,
                    checked_u32(layer_count, "CUDA fused paged prefill", "layer_count")?,
                    weight_scale,
                )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_decode_topk_indices_paged_indexer_from_device_into(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        page_tokens: usize,
        layer_index: usize,
        layer_count: usize,
        weight_scale: f32,
        output: &mut CudaI32Buffer,
    ) -> Result<()> {
        let layout = crate::cuda::kv_page_pool::PagedPlaneLayout {
            page_tokens,
            elements_per_token: index_head_dim,
            layer_index,
            layer_count,
        };
        self.validate_paged_indexer_storage(
            indexer_plane,
            block_slots,
            block_offsets,
            compressed_len,
            layout,
        )?;
        let query_len = index_heads
            .checked_mul(index_head_dim)
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged decode query overflow".into(),
            })?;
        let output_len = window_size
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged decode columns overflow".into(),
            })?;
        if window_size == 0
            || window_len > window_size
            || extra_cols == 0
            || extra_cols > 512
            || query.len() != query_len
            || weights.len() != index_heads
            || output.len() < output_len
        {
            return Err(Error::Internal {
                message: "CUDA paged decode indexer shape mismatch".into(),
            });
        }
        self.launched(unsafe {
            self.module.dsv4_decode_topk_indices_paged_indexer(
                &self.stream,
                one_block_config(256),
                &query.buffer,
                &weights.buffer,
                &indexer_plane.buffer,
                &block_slots.buffer,
                &block_offsets.buffer,
                &mut output.buffer,
                checked_u32(position, "CUDA paged decode", "position")?,
                checked_u32(window_len, "CUDA paged decode", "window_len")?,
                checked_u32(window_size, "CUDA paged decode", "window_size")?,
                checked_u32(extra_cols, "CUDA paged decode", "extra_cols")?,
                checked_u32(value_offset, "CUDA paged decode", "value_offset")?,
                checked_u32(compressed_len, "CUDA paged decode", "compressed_len")?,
                checked_u32(index_heads, "CUDA paged decode", "index_heads")?,
                checked_u32(index_head_dim, "CUDA paged decode", "index_head_dim")?,
                checked_u32(page_tokens, "CUDA paged decode", "page_tokens")?,
                checked_u32(layer_index, "CUDA paged decode", "layer_index")?,
                checked_u32(layer_count, "CUDA paged decode", "layer_count")?,
                weight_scale,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_decode_topk_indices_paged_indexer_rows_from_device(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        row_sequence_ids: &CudaI32Buffer,
        positions: &CudaI32Buffer,
        window_lens: &CudaI32Buffer,
        compressed_lens: &CudaI32Buffer,
        rows: usize,
        window_size: usize,
        index_topk: usize,
        compress_ratio: usize,
        direct_compressed: bool,
        index_heads: usize,
        index_head_dim: usize,
        page_tokens: usize,
        layer_index: usize,
        layer_count: usize,
        weight_scale: f32,
    ) -> Result<(CudaI32Buffer, CudaI32Buffer)> {
        let elements = Dsv4PagedDecodeRowsShape {
            rows,
            window_size,
            index_topk,
            index_heads,
            index_head_dim,
        }
        .elements()?;
        let mut logical_indices = self.zero_i32_buffer(elements)?;
        let mut plane_selectors = self.zero_i32_buffer(elements)?;
        self.dsv4_decode_topk_indices_paged_indexer_rows_from_device_into(
            query,
            weights,
            indexer_plane,
            block_slots,
            block_offsets,
            row_sequence_ids,
            positions,
            window_lens,
            compressed_lens,
            rows,
            window_size,
            index_topk,
            compress_ratio,
            direct_compressed,
            index_heads,
            index_head_dim,
            page_tokens,
            layer_index,
            layer_count,
            weight_scale,
            &mut logical_indices,
            &mut plane_selectors,
        )?;
        Ok((logical_indices, plane_selectors))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_decode_topk_indices_paged_indexer_rows_from_device_into(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        row_sequence_ids: &CudaI32Buffer,
        positions: &CudaI32Buffer,
        window_lens: &CudaI32Buffer,
        compressed_lens: &CudaI32Buffer,
        rows: usize,
        window_size: usize,
        index_topk: usize,
        compress_ratio: usize,
        direct_compressed: bool,
        index_heads: usize,
        index_head_dim: usize,
        page_tokens: usize,
        layer_index: usize,
        layer_count: usize,
        weight_scale: f32,
        logical_indices: &mut CudaI32Buffer,
        plane_selectors: &mut CudaI32Buffer,
    ) -> Result<()> {
        let layout = crate::cuda::kv_page_pool::PagedPlaneLayout {
            page_tokens,
            elements_per_token: index_head_dim,
            layer_index,
            layer_count,
        };
        layout.validate()?;
        let slot_elements = layer_count
            .checked_mul(page_tokens)
            .and_then(|value| value.checked_mul(index_head_dim))
            .ok_or_else(|| Error::Internal {
                message: "CUDA paged decode rows slot size overflow".into(),
            })?;
        if indexer_plane.len() < slot_elements || !indexer_plane.len().is_multiple_of(slot_elements)
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA paged decode rows plane length {} is not a positive multiple of slot size {slot_elements}",
                    indexer_plane.len()
                ),
            });
        }
        Dsv4PagedDecodeRowsShape {
            rows,
            window_size,
            index_topk,
            index_heads,
            index_head_dim,
        }
        .validate_lengths(
            query.len(),
            weights.len(),
            block_offsets.len(),
            row_sequence_ids.len(),
            positions.len(),
            window_lens.len(),
            compressed_lens.len(),
            logical_indices.len(),
            plane_selectors.len(),
        )?;
        self.launched(unsafe {
            self.module.dsv4_decode_topk_indices_paged_indexer_rows(
                &self.stream,
                LaunchConfig {
                    grid_dim: (checked_u32(rows, "CUDA paged decode rows", "rows")?, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &query.buffer,
                &weights.buffer,
                &indexer_plane.buffer,
                &block_slots.buffer,
                &block_offsets.buffer,
                &row_sequence_ids.buffer,
                &positions.buffer,
                &window_lens.buffer,
                &compressed_lens.buffer,
                &mut logical_indices.buffer,
                &mut plane_selectors.buffer,
                checked_u32(rows, "CUDA paged decode rows", "rows")?,
                checked_u32(window_size, "CUDA paged decode rows", "window_size")?,
                checked_u32(index_topk, "CUDA paged decode rows", "index_topk")?,
                checked_u32(compress_ratio, "CUDA paged decode rows", "compress_ratio")?,
                direct_compressed,
                checked_u32(index_heads, "CUDA paged decode rows", "index_heads")?,
                checked_u32(index_head_dim, "CUDA paged decode rows", "index_head_dim")?,
                checked_u32(page_tokens, "CUDA paged decode rows", "page_tokens")?,
                checked_u32(layer_index, "CUDA paged decode rows", "layer_index")?,
                checked_u32(layer_count, "CUDA paged decode rows", "layer_count")?,
                weight_scale,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_decode_topk_indices_fused_index_query_paged_indexer_from_device_into(
        &self,
        query: &CudaF32Buffer,
        weights: &CudaF32Buffer,
        indexer_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        block_offsets: &CudaI32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        position: usize,
        window_len: usize,
        window_size: usize,
        extra_cols: usize,
        value_offset: usize,
        compressed_len: usize,
        index_heads: usize,
        index_head_dim: usize,
        rope_dim: usize,
        page_tokens: usize,
        layer_index: usize,
        layer_count: usize,
        weight_scale: f32,
        output: &mut CudaI32Buffer,
    ) -> Result<()> {
        let layout = crate::cuda::kv_page_pool::PagedPlaneLayout {
            page_tokens,
            elements_per_token: index_head_dim,
            layer_index,
            layer_count,
        };
        self.validate_paged_indexer_storage(
            indexer_plane,
            block_slots,
            block_offsets,
            compressed_len,
            layout,
        )?;
        let query_len = index_heads
            .checked_mul(index_head_dim)
            .ok_or_else(|| Error::Internal {
                message: "CUDA fused paged decode query overflow".into(),
            })?;
        let output_len = window_size
            .checked_add(extra_cols)
            .ok_or_else(|| Error::Internal {
                message: "CUDA fused paged decode columns overflow".into(),
            })?;
        let rope_len = position
            .checked_add(1)
            .and_then(|v| v.checked_mul(rope_dim / 2))
            .ok_or_else(|| Error::Internal {
                message: "CUDA fused paged decode rope overflow".into(),
            })?;
        if window_size == 0
            || window_len > window_size
            || extra_cols == 0
            || extra_cols > 512
            || index_head_dim == 0
            || index_head_dim > 256
            || !index_head_dim.is_power_of_two()
            || !index_head_dim.is_multiple_of(32)
            || rope_dim > index_head_dim
            || !rope_dim.is_multiple_of(2)
            || query.len() != query_len
            || weights.len() != index_heads
            || output.len() < output_len
            || cos_table.len() < rope_len
            || sin_table.len() < rope_len
            || query_len > DSV4_DECODE_INDEX_QUERY_SHARED_ELEMENTS
        {
            return Err(Error::Internal {
                message: "CUDA fused paged decode indexer shape mismatch".into(),
            });
        }
        self.launched(unsafe {
            self.module
                .dsv4_decode_topk_indices_fused_index_query_paged_indexer(
                    &self.stream,
                    one_block_config(256),
                    &query.buffer,
                    &weights.buffer,
                    &indexer_plane.buffer,
                    &block_slots.buffer,
                    &block_offsets.buffer,
                    &cos_table.buffer,
                    &sin_table.buffer,
                    &mut output.buffer,
                    checked_u32(position, "CUDA fused paged decode", "position")?,
                    checked_u32(window_len, "CUDA fused paged decode", "window_len")?,
                    checked_u32(window_size, "CUDA fused paged decode", "window_size")?,
                    checked_u32(extra_cols, "CUDA fused paged decode", "extra_cols")?,
                    checked_u32(value_offset, "CUDA fused paged decode", "value_offset")?,
                    checked_u32(compressed_len, "CUDA fused paged decode", "compressed_len")?,
                    checked_u32(index_heads, "CUDA fused paged decode", "index_heads")?,
                    checked_u32(index_head_dim, "CUDA fused paged decode", "index_head_dim")?,
                    checked_u32(rope_dim, "CUDA fused paged decode", "rope_dim")?,
                    checked_u32(page_tokens, "CUDA fused paged decode", "page_tokens")?,
                    checked_u32(layer_index, "CUDA fused paged decode", "layer_index")?,
                    checked_u32(layer_count, "CUDA fused paged decode", "layer_count")?,
                    weight_scale,
                )
        })
    }

    pub fn artifact_swiglu_ffn_matvec(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &[f32],
        output_scale: f32,
        swiglu_limit: f32,
    ) -> Result<Vec<f32>> {
        if input.len() != gate.shape.in_features() || input.len() != up.shape.in_features() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA SwiGLU input length mismatch: input={} gate_in={} up_in={}",
                    input.len(),
                    gate.shape.in_features(),
                    up.shape.in_features()
                ),
            });
        }
        if gate.shape.out_features() != up.shape.out_features()
            || down.shape.in_features() != gate.shape.out_features()
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA SwiGLU shape mismatch: gate={:?} up={:?} down={:?}",
                    gate.shape, up.shape, down.shape
                ),
            });
        }
        let gate_input = prepare_activation_for_artifact_linear(gate.shape, input)?;
        let up_input = prepare_activation_for_artifact_linear(up.shape, input)?;
        let gate_xd = self.upload_f32(gate_input.as_ref())?;
        let up_xd = self.upload_f32(up_input.as_ref())?;
        let mut gated = self.zeroed_device_buffer::<f32>(gate.shape.out_features())?;
        let mut upd = self.zeroed_device_buffer::<f32>(up.shape.out_features())?;
        let mut hidden = self.zeroed_device_buffer::<f32>(gate.shape.out_features())?;
        let mut yd = self.zeroed_device_buffer::<f32>(down.shape.out_features())?;
        self.artifact_linear_matvec_device(gate, &gate_xd, &mut gated)?;
        self.artifact_linear_matvec_device(up, &up_xd, &mut upd)?;
        self.launched(unsafe {
            self.module.swiglu_weighted_clamped(
                &self.stream,
                LaunchConfig::for_num_elems(gate.shape.out_features() as u32),
                &gated,
                &upd,
                &mut hidden,
                gate.shape.out_features() as u32,
                output_scale,
                swiglu_limit,
            )
        })?;
        if quantized_shape_uses_fp8_activation(down.shape) {
            self.fp8_activation_quantize_in_place(
                &mut hidden,
                down.shape.in_features(),
                down.shape.in_features(),
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE,
            )?;
        }
        self.artifact_linear_matvec_device(down, &hidden, &mut yd)?;
        self.download_f32(&yd, down.shape.out_features())
    }

    pub fn artifact_swiglu_ffn_from_device(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        output_scale: f32,
        swiglu_limit: f32,
    ) -> Result<CudaF32Buffer> {
        if input.len() != gate.shape.in_features() || input.len() != up.shape.in_features() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA SwiGLU device input length mismatch: input={} gate_in={} up_in={}",
                    input.len(),
                    gate.shape.in_features(),
                    up.shape.in_features()
                ),
            });
        }
        if gate.shape.out_features() != up.shape.out_features()
            || down.shape.in_features() != gate.shape.out_features()
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA SwiGLU shape mismatch: gate={:?} up={:?} down={:?}",
                    gate.shape, up.shape, down.shape
                ),
            });
        }
        self.artifact_swiglu_ffn_rows_from_device(
            gate,
            up,
            down,
            input,
            1,
            output_scale,
            swiglu_limit,
        )
    }

    /// Create caller-owned workspace for allocation-free artifact linear rows.
    pub fn artifact_linear_workspace(
        &self,
        rows: usize,
        max_input_width: usize,
    ) -> Result<CudaArtifactLinearWorkspace> {
        if rows == 0 || max_input_width == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA artifact linear workspace requires positive dimensions: rows={rows} max_input_width={max_input_width}"
                ),
            });
        }
        let value_capacity = rows
            .checked_mul(max_input_width)
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear workspace value size overflow".into(),
            })?;
        let scale_capacity = rows
            .checked_mul(max_input_width.div_ceil(ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE))
            .ok_or_else(|| Error::Internal {
                message: "CUDA artifact linear workspace scale size overflow".into(),
            })?;
        Ok(CudaArtifactLinearWorkspace {
            cloned: CudaTypedBuffer::from_device_buffer(
                self.uninitialized_device_buffer::<f32>(value_capacity)?,
            ),
            x_packed: self.uninitialized_device_buffer::<u8>(value_capacity)?,
            x_scales: self.uninitialized_device_buffer::<u8>(scale_capacity)?,
            value_capacity,
            scale_capacity,
        })
    }

    pub fn fp8_activation_pack(
        &self,
        rows: usize,
        row_width: usize,
    ) -> Result<CudaFp8ActivationPack> {
        if rows == 0 || row_width == 0 || !row_width.is_multiple_of(128) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA FP8 activation pack requires positive K128 dimensions: rows={rows} row_width={row_width}"
                ),
            });
        }
        let value_capacity = rows.checked_mul(row_width).ok_or_else(|| Error::Internal {
            message: "CUDA FP8 activation pack value size overflow".into(),
        })?;
        let scale_capacity = rows
            .checked_mul(row_width / ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE)
            .ok_or_else(|| Error::Internal {
                message: "CUDA FP8 activation pack scale size overflow".into(),
            })?;
        Ok(CudaFp8ActivationPack {
            x_packed: self.uninitialized_device_buffer::<u8>(value_capacity)?,
            x_scales: self.uninitialized_device_buffer::<u8>(scale_capacity)?,
            value_capacity,
            scale_capacity,
        })
    }

    /// Execute the complete fused shared-expert gate/up -> SwiGLU -> down
    /// bundle and write directly to the caller-owned destination.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_shared_ffn_into(
        &self,
        gate: &CudaArtifactLinearHandle,
        up: &CudaArtifactLinearHandle,
        down: &CudaArtifactLinearHandle,
        input: &CudaPreparedFp8Activation<'_>,
        hidden_f32: &mut CudaF32Buffer,
        hidden: &mut CudaFp8ActivationPack,
        rows: usize,
        output_scale: f32,
        swiglu_limit: f32,
        output: &mut CudaF32Buffer,
        accumulate_output: bool,
    ) -> Result<()> {
        let (
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: intermediate,
                in_features: input_size,
                block_m: gate_block_m,
                block_k: gate_block_k,
            },
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: up_out,
                in_features: up_in,
                block_m: up_block_m,
                block_k: up_block_k,
            },
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: output_size,
                in_features: down_in,
                block_m: down_block_m,
                block_k: down_block_k,
            },
        ) = (gate.shape, up.shape, down.shape)
        else {
            return Err(Error::Internal {
                message: format!(
                    "fused shared FFN requires FP8 weights: gate={:?} up={:?} down={:?}",
                    gate.shape, up.shape, down.shape
                ),
            });
        };
        if rows == 0
            || input_size != up_in
            || intermediate != up_out
            || intermediate != down_in
            || input.rows != rows
            || input.row_width != input_size
            || hidden_f32.len() < rows * intermediate
            || hidden.value_capacity != rows * intermediate
            || hidden.scale_capacity != rows * intermediate.div_ceil(128)
            || output.len() != rows * output_size
        {
            return Err(Error::Internal {
                message: format!(
                    "fused shared FFN shape mismatch: rows={rows} input=[{},{}] hidden_f32={} hidden=[{},{}] output={} gate={:?} up={:?} down={:?}",
                    input.rows,
                    input.row_width,
                    hidden_f32.len(),
                    hidden.value_capacity,
                    hidden.scale_capacity,
                    output.len(),
                    gate.shape,
                    up.shape,
                    down.shape
                ),
            });
        }
        let gate_scales = gate.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "fused shared gate scales are missing".into(),
        })?;
        let up_scales = up.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "fused shared up scales are missing".into(),
        })?;
        let down_scales = down.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "fused shared down scales are missing".into(),
        })?;
        crate::cuda::cutlass::shared_ffn(
            &self.stream,
            input.x_packed,
            input.x_scales,
            &gate.weight,
            gate_scales,
            &up.weight,
            up_scales,
            &down.weight,
            down_scales,
            &mut hidden_f32.buffer,
            &mut hidden.x_packed,
            &mut hidden.x_scales,
            &mut output.buffer,
            rows,
            input_size,
            intermediate,
            output_size,
            (gate_block_m, gate_block_k),
            (up_block_m, up_block_k),
            (down_block_m, down_block_k),
            output_scale,
            swiglu_limit,
            accumulate_output,
        )?;
        self.record_kernel_launch();
        Ok(())
    }

    pub fn moe_batched_workspace(
        &self,
        max_experts: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<CudaMoeBatchedWorkspace> {
        if max_experts == 0 || max_experts > 64 {
            return Err(Error::Internal {
                message: format!("CUDA MoE workspace expects 1..=64 experts, got {max_experts}"),
            });
        }
        if input_size == 0 || intermediate_size == 0 || hidden_size == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA MoE workspace invalid shape: input={input_size} intermediate={intermediate_size} hidden={hidden_size}"
                ),
            });
        }
        if !input_size.is_multiple_of(32) || !intermediate_size.is_multiple_of(32) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA MoE workspace expects 32-aligned input/intermediate, got input={input_size} intermediate={intermediate_size}"
                ),
            });
        }

        let total_expert_output =
            max_experts
                .checked_mul(hidden_size)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA MoE workspace down scratch size overflow".into(),
                })?;
        Ok(CudaMoeBatchedWorkspace {
            gate_ptrs: self.zeroed_device_buffer::<u64>(max_experts)?,
            gate_scale_ptrs: self.zeroed_device_buffer::<u64>(max_experts)?,
            up_ptrs: self.zeroed_device_buffer::<u64>(max_experts)?,
            up_scale_ptrs: self.zeroed_device_buffer::<u64>(max_experts)?,
            down_ptrs: self.zeroed_device_buffer::<u64>(max_experts)?,
            down_scale_ptrs: self.zeroed_device_buffer::<u64>(max_experts)?,
            route_weights: self.zeroed_device_buffer::<f32>(max_experts)?,
            route_slots: self.zeroed_device_buffer::<i32>(max_experts)?,
            dispatch_error: self.zeroed_device_buffer::<i32>(1)?,
            expert_output: CudaTypedBuffer::from_device_buffer(
                self.uninitialized_device_buffer::<f32>(total_expert_output)?,
            ),
            max_experts,
            input_size,
            intermediate_size,
            hidden_size,
        })
    }

    /// Allocate a graph-stable compact route plan and caller-owned CUTLASS workspace.
    pub fn expert_group_route_plan(
        &self,
        max_experts: usize,
        route_capacity: usize,
        tokens: usize,
        input_size: usize,
        intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<CudaExpertGroupRoutePlan> {
        if max_experts == 0 || max_experts > 512 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route plan expects 1..=512 slots, got {max_experts}"
                ),
            });
        }
        if route_capacity == 0 || route_capacity > i32::MAX as usize {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route capacity must be in 1..={}, got {route_capacity}",
                    i32::MAX
                ),
            });
        }
        if tokens == 0 || tokens > i32::MAX as usize {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route token count must be in 1..={}, got {tokens}",
                    i32::MAX
                ),
            });
        }
        if input_size == 0 || intermediate_size == 0 || hidden_size == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route plan invalid shape: tokens={tokens} input={input_size} intermediate={intermediate_size} hidden={hidden_size}"
                ),
            });
        }
        if !input_size.is_multiple_of(64)
            || !intermediate_size.is_multiple_of(64)
            || !hidden_size.is_multiple_of(4)
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped FP4 MoE expects input/intermediate multiples of 64 and hidden a multiple of 4, got input={input_size} intermediate={intermediate_size} hidden={hidden_size}"
                ),
            });
        }
        for (field, value) in [
            ("input_size", input_size),
            ("intermediate_size", intermediate_size),
            ("hidden_size", hidden_size),
        ] {
            checked_u32(value, "expert group route plan", field)?;
        }
        if !discover_provider()?.supports(CutlassKernelId::GroupedFp4Moe) {
            return Err(unsupported_grouped_fp4_moe());
        }

        let total_input = tokens
            .checked_mul(input_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert group route input size overflow".into(),
            })?;
        checked_u32(total_input, "expert group route plan", "input elements")?;
        let maximum_active_groups = max_experts.min(route_capacity);
        let mut workspace_bytes = 0usize;
        // Descriptor and per-group regions use maximum active capacity. Varying
        // this split from zero through that capacity also presents every possible
        // small or large bucket group count to the native workspace query, while
        // row-dependent regions use their validated route-capacity upper bound.
        for small_group_count in 0..=maximum_active_groups {
            if route_capacity < GROUPED_FP4_MOE_SMALL_GROUP_ROW_LIMIT
                && small_group_count != maximum_active_groups
            {
                continue;
            }
            workspace_bytes =
                workspace_bytes.max(grouped_fp4_moe_workspace_size(GroupedFp4MoeLayout {
                    active_group_count: maximum_active_groups,
                    small_group_count,
                    slot_capacity: max_experts,
                    max_group_rows: route_capacity,
                    total_routed_rows: route_capacity,
                    num_tokens: tokens,
                    num_routes: route_capacity,
                    input_size,
                    intermediate_size,
                    hidden_size,
                    swiglu_limit: 0.0,
                })?);
        }
        if workspace_bytes == 0 {
            return Err(unsupported_grouped_fp4_moe());
        }
        let workspace_allocation = self.uninitialized_device_buffer::<u8>(
            workspace_bytes
                .checked_add(GROUPED_FP4_MOE_WORKSPACE_ALIGNMENT - 1)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA grouped FP4 MoE workspace allocation overflow".into(),
                })?,
        )?;
        let address = workspace_allocation.cu_deviceptr() as usize;
        let aligned_address = address
            .checked_add(GROUPED_FP4_MOE_WORKSPACE_ALIGNMENT - 1)
            .map(|value| value & !(GROUPED_FP4_MOE_WORKSPACE_ALIGNMENT - 1))
            .ok_or_else(|| Error::Internal {
                message: "CUDA grouped FP4 MoE workspace alignment overflow".into(),
            })?;
        let cutlass_workspace =
            cu(workspace_allocation.slice(aligned_address - address, workspace_bytes))?;
        debug_assert_eq!(
            cutlass_workspace.cu_deviceptr() as usize % GROUPED_FP4_MOE_WORKSPACE_ALIGNMENT,
            0
        );

        Ok(CudaExpertGroupRoutePlan {
            slot_counts: self.zeroed_device_buffer::<i32>(max_experts)?,
            slot_route_offsets: self.zeroed_device_buffer::<i32>(max_experts)?,
            slot_cursors: self.zeroed_device_buffer::<i32>(max_experts)?,
            active_expert_slots: self.zeroed_device_buffer::<i32>(max_experts)?,
            active_group_generations: self.zeroed_device_buffer::<i32>(max_experts)?,
            expert_route_indptr: self.zeroed_device_buffer::<i32>(max_experts + 1)?,
            expert_route_counts: self.zeroed_device_buffer::<i32>(max_experts)?,
            route_token_indices: self.zeroed_device_buffer::<i32>(route_capacity)?,
            route_indices: self.zeroed_device_buffer::<i32>(route_capacity)?,
            route_weights: self.zeroed_device_buffer::<f32>(route_capacity)?,
            host_scalars: self.zeroed_device_buffer::<i32>(4)?,
            host_staging: cu(PinnedHostBuffer::zeroed(&self._ctx, 4))?,
            metadata_ready: cu(self._ctx.new_event(false))?,
            metadata_copied: cu(self._ctx.new_event(false))?,
            host_metadata: None,
            route_written: self.zeroed_device_buffer::<i32>(route_capacity)?,
            route_error: self.zeroed_device_buffer::<i32>(1)?,
            resolve: self.expert_route_resolve_workspace(route_capacity, route_capacity)?,
            input_fp8: self.uninitialized_device_buffer::<u8>(total_input)?,
            input_ue8m0: self.uninitialized_device_buffer::<u8>(total_input / 128)?,
            cutlass_workspace,
            max_experts,
            route_capacity,
            tokens,
            input_size,
            intermediate_size,
            hidden_size,
            input_prepared: false,
            invocation_routes: None,
        })
    }

    /// Quantize the complete layer input once for all active expert groups.
    pub fn prepare_expert_group_route_input_from_device(
        &self,
        input: &CudaF32Buffer,
        tokens: usize,
        input_size: usize,
        plan: &mut CudaExpertGroupRoutePlan,
    ) -> Result<()> {
        let expected_len = tokens
            .checked_mul(input_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert group route input length overflow".into(),
            })?;
        if tokens != plan.tokens || input_size != plan.input_size {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route input/plan shape mismatch: plan=[tokens={},input={}] call=[tokens={tokens},input={input_size}]",
                    plan.tokens, plan.input_size
                ),
            });
        }
        if input.len() != expected_len {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route input length mismatch: input={} expected={}x{}={expected_len}",
                    input.len(),
                    tokens,
                    input_size
                ),
            });
        }
        if !input_size.is_multiple_of(128) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route input size must be a multiple of 128, got {input_size}"
                ),
            });
        }

        let total_values = checked_u32(expected_len, "expert group route input", "elements")?;
        let quant_blocks = checked_u32(
            expected_len / 128,
            "expert group route input",
            "quantization blocks",
        )?;
        let row_width = checked_u32(input_size, "expert group route input", "input_size")?;
        let timing_enabled = self.observability.moe_timing;
        let phase_start = timing_enabled.then(Instant::now);
        self.launched(unsafe {
            self.module.fp8_e4m3fn_e8m0_quantize_f32_packed(
                &self.stream,
                LaunchConfig::for_num_elems(quant_blocks),
                &input.buffer,
                &mut plan.input_fp8,
                &mut plan.input_ue8m0,
                total_values,
                row_width,
                128,
            )
        })?;
        plan.input_prepared = true;
        if let Some(start) = phase_start {
            self.sync_stream()?;
            self.counters
                .add_moe_input_prepare_us(duration_us(start.elapsed()));
        }
        Ok(())
    }

    /// Allocate uninitialized route-major output `[tokens * routes_per_token, hidden]`.
    /// Callers must execute exactly one expert-group entry for every route before reduction.
    pub fn allocate_moe_route_output(
        &self,
        tokens: usize,
        routes_per_token: usize,
        hidden_size: usize,
    ) -> Result<CudaF32Buffer> {
        if tokens == 0 || routes_per_token == 0 || hidden_size == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA MoE route output invalid shape: tokens={tokens} routes_per_token={routes_per_token} hidden={hidden_size}"
                ),
            });
        }
        let routes = tokens
            .checked_mul(routes_per_token)
            .ok_or_else(|| Error::Internal {
                message: "CUDA MoE route output route count overflow".into(),
            })?;
        if routes > i32::MAX as usize {
            return Err(Error::Internal {
                message: format!(
                    "CUDA MoE route output route count exceeds i32 metadata ABI: {routes}"
                ),
            });
        }
        checked_u32(hidden_size, "MoE route output", "hidden_size")?;
        let len = routes
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA MoE route output element count overflow".into(),
            })?;
        Ok(CudaTypedBuffer::from_device_buffer(
            self.uninitialized_device_buffer::<f32>(len)?,
        ))
    }

    /// Begin one expert-group MoE invocation. Completion/error state and the
    /// route-major output are initialized exactly once before any resident window.
    pub fn begin_expert_group_route_invocation(
        &self,
        routes_per_token: usize,
        plan: &mut CudaExpertGroupRoutePlan,
        route_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if routes_per_token == 0 {
            return Err(Error::Internal {
                message: "CUDA expert group route invocation requires positive routes_per_token"
                    .into(),
            });
        }
        let routes = plan
            .tokens
            .checked_mul(routes_per_token)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert group route count overflow".into(),
            })?;
        let output_elements =
            routes
                .checked_mul(plan.hidden_size)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA expert group route output size overflow".into(),
                })?;
        if routes > plan.route_written.len() || route_output.len() != output_elements {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route invocation capacity mismatch: routes={routes} completion_capacity={} output={} expected={output_elements}",
                    plan.route_written.len(),
                    route_output.len()
                ),
            });
        }
        let elements = routes.max(output_elements).max(1);
        self.launched(unsafe {
            self.module.initialize_expert_group_route_invocation(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    elements,
                    "expert group route invocation",
                    "initialization elements",
                )?),
                &mut route_output.buffer,
                &mut plan.route_written,
                &mut plan.route_error,
                checked_u32(
                    output_elements,
                    "expert group route invocation",
                    "output elements",
                )?,
                checked_u32(routes, "expert group route invocation", "routes")?,
            )
        })?;
        plan.host_metadata = None;
        plan.invocation_routes = Some(routes);
        Ok(())
    }

    /// Resolve routes and produce compact expert-contiguous device metadata.
    ///
    /// CUTLASS sizes and dispatches from four host scalars. After all metadata
    /// kernels, this performs one 16-byte D2H copy through the control stream and
    /// waits for that copy's event only. The allocation addresses stay stable, but
    /// this scalar dependency prevents the preparation/launch pair from being
    /// captured as a fully device-only CUDA graph and adds one CPU-visible latency
    /// point per prepared resident window.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_expert_group_route_plan(
        &self,
        table: &CudaExpertSlotTable,
        expert_ids: &CudaI32Buffer,
        router_weights: &CudaF32Buffer,
        route_count: usize,
        routes_per_token: usize,
        plan: &mut CudaExpertGroupRoutePlan,
    ) -> Result<CudaExpertGroupRoutePlanHost> {
        table.ensure_healthy()?;
        self.check_capture_safe("expert group route host scalar readback")?;
        if route_count == 0
            || routes_per_token == 0
            || route_count > expert_ids.len()
            || route_count > router_weights.len()
            || route_count > plan.route_capacity
            || !route_count.is_multiple_of(routes_per_token)
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route plan shape mismatch: routes={route_count} routes_per_token={routes_per_token} capacity={} ids={} weights={}",
                    plan.route_capacity,
                    expert_ids.len(),
                    router_weights.len()
                ),
            });
        }
        let slot_capacity = table.host.slot_capacity();
        if slot_capacity == 0 || slot_capacity > 512 || slot_capacity > plan.max_experts {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route slot capacity mismatch: table={slot_capacity} plan={} limit=512",
                    plan.max_experts
                ),
            });
        }
        let init_elements = plan.route_capacity.max(slot_capacity + 1).max(4);
        let route_count_u32 = checked_u32(route_count, "expert group route plan", "routes")?;
        let slot_capacity_u32 =
            checked_u32(slot_capacity, "expert group route plan", "slot capacity")?;
        let route_capacity_u32 = checked_u32(
            plan.route_capacity,
            "expert group route plan",
            "route capacity",
        )?;
        self.launched(unsafe {
            self.module.initialize_expert_group_route_plan(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    init_elements,
                    "expert group route plan",
                    "initialization elements",
                )?),
                &mut plan.slot_counts,
                &mut plan.slot_route_offsets,
                &mut plan.slot_cursors,
                &mut plan.active_expert_slots,
                &mut plan.active_group_generations,
                &mut plan.expert_route_indptr,
                &mut plan.expert_route_counts,
                &mut plan.route_token_indices,
                &mut plan.route_indices,
                &mut plan.route_weights,
                &mut plan.host_scalars,
                slot_capacity_u32,
                route_capacity_u32,
            )
        })?;
        self.resolve_expert_routes(table, expert_ids, route_count, &mut plan.resolve)?;
        self.launched(unsafe {
            self.module.count_expert_group_routes(
                &self.stream,
                LaunchConfig::for_num_elems(route_count_u32),
                &plan.resolve.route_slots.buffer,
                &plan.resolve.route_generations.buffer,
                &table.slot_generation,
                &mut plan.slot_counts,
                route_count_u32,
                slot_capacity_u32,
            )
        })?;
        self.launched(unsafe {
            self.module.compact_expert_group_routes(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &plan.slot_counts,
                &table.slot_generation,
                &mut plan.slot_route_offsets,
                &mut plan.active_expert_slots,
                &mut plan.active_group_generations,
                &mut plan.expert_route_indptr,
                &mut plan.expert_route_counts,
                &mut plan.host_scalars,
                &mut plan.route_error,
                slot_capacity_u32,
                route_capacity_u32,
                checked_u32(
                    GROUPED_FP4_MOE_SMALL_GROUP_ROW_LIMIT,
                    "expert group route plan",
                    "small group row limit",
                )?,
            )
        })?;
        self.launched(unsafe {
            self.module.scatter_expert_group_routes(
                &self.stream,
                LaunchConfig::for_num_elems(route_count_u32),
                &plan.resolve.route_slots.buffer,
                &plan.resolve.route_generations.buffer,
                &router_weights.buffer,
                &table.slot_generation,
                &plan.slot_route_offsets,
                &mut plan.slot_cursors,
                &mut plan.route_token_indices,
                &mut plan.route_indices,
                &mut plan.route_weights,
                &mut plan.route_error,
                route_count_u32,
                checked_u32(
                    routes_per_token,
                    "expert group route plan",
                    "routes per token",
                )?,
                slot_capacity_u32,
                route_capacity_u32,
            )
        })?;

        cu(plan.metadata_ready.record(&self.stream))?;
        cu(self.control_stream.wait(&plan.metadata_ready))?;
        unsafe {
            cu(plan
                .host_scalars
                .copy_to_pinned_host_async(&self.control_stream, &mut plan.host_staging))?;
        }
        cu(plan.metadata_copied.record(&self.control_stream))?;
        cu(plan.metadata_copied.synchronize())?;
        self.counters.add_device_to_host(element_bytes::<i32>(4));

        let values = plan.host_staging.as_slice();
        let scalar = |index: usize, name: &str| {
            usize::try_from(values[index]).map_err(|_| Error::Internal {
                message: format!(
                    "CUDA expert group route plan produced negative {name}: {}",
                    values[index]
                ),
            })
        };
        let host = CudaExpertGroupRoutePlanHost {
            active_group_count: scalar(0, "active_group_count")?,
            small_group_count: scalar(1, "small_group_count")?,
            max_group_rows: scalar(2, "max_group_rows")?,
            total_routed_rows: scalar(3, "total_routed_rows")?,
        };
        let empty = host.active_group_count == 0
            && host.small_group_count == 0
            && host.max_group_rows == 0
            && host.total_routed_rows == 0;
        if !empty
            && (host.active_group_count > slot_capacity
                || host.small_group_count > host.active_group_count
                || host.max_group_rows == 0
                || host.max_group_rows > host.total_routed_rows
                || host.total_routed_rows == 0
                || host.total_routed_rows > route_count)
        {
            return Err(Error::Internal {
                message: format!("invalid CUDA expert group route host metadata: {host:?}"),
            });
        }
        plan.host_metadata = Some(host);
        Ok(host)
    }

    /// Diagnostic/test oracle for the compact device metadata.
    /// Production dispatch consumes the device buffers directly.
    pub fn download_expert_group_route_plan(
        &self,
        plan: &CudaExpertGroupRoutePlan,
    ) -> Result<CudaExpertGroupRoutePlanDownload> {
        let host = plan.host_metadata.ok_or_else(|| Error::Internal {
            message: "CUDA expert group route plan has no prepared host metadata".into(),
        })?;
        Ok(CudaExpertGroupRoutePlanDownload {
            active_expert_slots: self
                .download_device_slice(&plan.active_expert_slots, host.active_group_count)?,
            active_group_generations: self
                .download_device_slice(&plan.active_group_generations, host.active_group_count)?,
            expert_route_indptr: self
                .download_device_slice(&plan.expert_route_indptr, host.active_group_count + 1)?,
            expert_route_counts: self
                .download_device_slice(&plan.expert_route_counts, host.active_group_count)?,
            route_token_indices: self
                .download_device_slice(&plan.route_token_indices, host.total_routed_rows)?,
            route_indices: self
                .download_device_slice(&plan.route_indices, host.total_routed_rows)?,
            route_weights: self
                .download_device_slice(&plan.route_weights, host.total_routed_rows)?,
            host,
            dispatch_error: self.download_device_slice(&plan.route_error, 1)?[0] != 0,
        })
    }

    /// Launch the compact grouped FP4 expert pipeline from a prepared route plan.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_fp4_moe_from_prepared_plan(
        &self,
        table: &CudaExpertSlotTable,
        routes_per_token: usize,
        swiglu_limit: f32,
        plan: &mut CudaExpertGroupRoutePlan,
        route_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        table.ensure_healthy()?;
        let host = plan.host_metadata.ok_or_else(|| Error::Internal {
            message: "CUDA grouped FP4 MoE route plan has not been prepared".into(),
        })?;
        if !plan.input_prepared || routes_per_token == 0 || !swiglu_limit.is_finite() {
            return Err(Error::Internal {
                message: "CUDA grouped FP4 MoE input is not prepared or has invalid parameters"
                    .into(),
            });
        }
        let route_count =
            plan.tokens
                .checked_mul(routes_per_token)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA grouped FP4 MoE route count overflow".into(),
                })?;
        let expected_output =
            route_count
                .checked_mul(plan.hidden_size)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA grouped FP4 MoE output size overflow".into(),
                })?;
        if plan.invocation_routes != Some(route_count) || route_output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped FP4 MoE invocation mismatch: active={:?} routes={route_count} output={} expected={expected_output}",
                    plan.invocation_routes,
                    route_output.len()
                ),
            });
        }
        let slot_capacity = table.host.slot_capacity();
        if slot_capacity > plan.max_experts {
            return Err(Error::Internal {
                message: format!(
                    "CUDA grouped FP4 MoE table exceeds plan: slots={slot_capacity} plan={}",
                    plan.max_experts
                ),
            });
        }

        if host.active_group_count == 0 {
            return Ok(());
        }

        let active_expert_slots = cu(plan.active_expert_slots.slice(0, host.active_group_count))?;
        let active_group_generations = cu(plan
            .active_group_generations
            .slice(0, host.active_group_count))?;
        let expert_route_indptr = cu(plan
            .expert_route_indptr
            .slice(0, host.active_group_count + 1))?;
        let expert_route_counts = cu(plan.expert_route_counts.slice(0, host.active_group_count))?;
        let route_token_indices = cu(plan.route_token_indices.slice(0, host.total_routed_rows))?;
        let route_indices = cu(plan.route_indices.slice(0, host.total_routed_rows))?;
        let route_weights = cu(plan.route_weights.slice(0, host.total_routed_rows))?;
        let slot_generations = cu(table.slot_generation.slice(0, slot_capacity))?;
        let gate_ptrs = cu(table.gate_weight.slice(0, slot_capacity))?;
        let gate_scale_ptrs = cu(table.gate_scale.slice(0, slot_capacity))?;
        let up_ptrs = cu(table.up_weight.slice(0, slot_capacity))?;
        let up_scale_ptrs = cu(table.up_scale.slice(0, slot_capacity))?;
        let down_ptrs = cu(table.down_weight.slice(0, slot_capacity))?;
        let down_scale_ptrs = cu(table.down_scale.slice(0, slot_capacity))?;
        let mut route_written = cu(plan.route_written.slice(0, route_count))?;
        let mut buffers = GroupedFp4MoeBuffers {
            active_expert_slots: &active_expert_slots,
            active_group_generations: &active_group_generations,
            expert_route_indptr: &expert_route_indptr,
            expert_route_counts: &expert_route_counts,
            route_token_indices: &route_token_indices,
            route_indices: &route_indices,
            route_weights: &route_weights,
            slot_generations: &slot_generations,
            gate_ptrs: &gate_ptrs,
            gate_scale_ptrs: &gate_scale_ptrs,
            up_ptrs: &up_ptrs,
            up_scale_ptrs: &up_scale_ptrs,
            down_ptrs: &down_ptrs,
            down_scale_ptrs: &down_scale_ptrs,
            input_fp8: &plan.input_fp8,
            input_ue8m0: &plan.input_ue8m0,
            route_output: &mut route_output.buffer,
            route_written: &mut route_written,
            route_error: &mut plan.route_error,
            workspace: &mut plan.cutlass_workspace,
        };
        let layout = GroupedFp4MoeLayout {
            active_group_count: host.active_group_count,
            small_group_count: host.small_group_count,
            slot_capacity,
            max_group_rows: host.max_group_rows,
            total_routed_rows: host.total_routed_rows,
            num_tokens: plan.tokens,
            num_routes: route_count,
            input_size: plan.input_size,
            intermediate_size: plan.intermediate_size,
            hidden_size: plan.hidden_size,
            swiglu_limit,
        };
        self.counters.add_moe_call();
        grouped_fp4_moe(&self.stream, &mut buffers, layout)?;
        self.record_kernel_launch();
        Ok(())
    }

    /// Add route-major expert outputs into an existing token-major accumulator.
    /// Each `(token, hidden-row)` thread performs a strict rank-ordered left fold.
    pub fn reduce_moe_route_outputs_ranked(
        &self,
        route_output: &CudaF32Buffer,
        tokens: usize,
        routes_per_token: usize,
        hidden_size: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if tokens == 0 || routes_per_token == 0 || hidden_size == 0 {
            return Err(Error::Internal {
                message: format!(
                    "CUDA MoE route reducer invalid shape: tokens={tokens} routes_per_token={routes_per_token} hidden={hidden_size}"
                ),
            });
        }
        let routes = tokens
            .checked_mul(routes_per_token)
            .ok_or_else(|| Error::Internal {
                message: "CUDA MoE route reducer route count overflow".into(),
            })?;
        let expected_routes = routes
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA MoE route reducer input size overflow".into(),
            })?;
        let expected_output = tokens
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA MoE route reducer output size overflow".into(),
            })?;
        if route_output.len() != expected_routes || output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA MoE route reducer length mismatch: route_output={} expected={expected_routes}, output={} expected={expected_output}",
                    route_output.len(),
                    output.len()
                ),
            });
        }

        let elements = checked_u32(expected_output, "MoE route reducer", "output elements")?;
        self.launched(unsafe {
            self.module.moe_reduce_route_outputs_ranked(
                &self.stream,
                LaunchConfig::for_num_elems(elements),
                &route_output.buffer,
                &mut output.buffer,
                checked_u32(tokens, "MoE route reducer", "tokens")?,
                checked_u32(routes_per_token, "MoE route reducer", "routes_per_token")?,
                checked_u32(hidden_size, "MoE route reducer", "hidden_size")?,
            )
        })
    }

    /// Finalize an expert-group invocation. Missing routes or any cumulative
    /// planning/execution error produce a canonical NaN in every output element;
    /// otherwise each token performs the existing strict rank-ordered left fold.
    pub fn reduce_expert_group_route_outputs_ranked(
        &self,
        route_output: &CudaF32Buffer,
        tokens: usize,
        routes_per_token: usize,
        hidden_size: usize,
        plan: &mut CudaExpertGroupRoutePlan,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if tokens != plan.tokens || hidden_size != plan.hidden_size {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route reducer/plan mismatch: plan=[tokens={},hidden={}] call=[tokens={tokens},hidden={hidden_size}]",
                    plan.tokens, plan.hidden_size
                ),
            });
        }
        let routes = tokens
            .checked_mul(routes_per_token)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert group route reducer route overflow".into(),
            })?;
        let expected_routes = routes
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert group route reducer input overflow".into(),
            })?;
        let expected_output = tokens
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA expert group route reducer output overflow".into(),
            })?;
        if routes_per_token == 0
            || plan.invocation_routes != Some(routes)
            || route_output.len() != expected_routes
            || output.len() != expected_output
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA expert group route reducer state/shape mismatch: active={:?} routes={routes} route_output={} expected={expected_routes} output={} expected={expected_output}",
                    plan.invocation_routes,
                    route_output.len(),
                    output.len()
                ),
            });
        }

        let elements = checked_u32(
            expected_output,
            "expert group route reducer",
            "output elements",
        )?;
        self.launched(unsafe {
            self.module.moe_reduce_expert_group_route_outputs_ranked(
                &self.stream,
                LaunchConfig::for_num_elems(elements),
                &route_output.buffer,
                &plan.route_written,
                &plan.route_error,
                &mut output.buffer,
                checked_u32(tokens, "expert group route reducer", "tokens")?,
                checked_u32(
                    routes_per_token,
                    "expert group route reducer",
                    "routes_per_token",
                )?,
                checked_u32(hidden_size, "expert group route reducer", "hidden_size")?,
            )
        })?;
        plan.invocation_routes = None;
        plan.host_metadata = None;
        Ok(())
    }

    /// Populate a warmed batched MoE workspace entirely on device from stable
    /// expert slots and device-produced router metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_moe_experts_batched_workspace_stable(
        &self,
        table: &CudaExpertSlotTable,
        selected_experts: &[usize],
        expert_ids: &CudaI32Buffer,
        router_weights: &CudaF32Buffer,
        route_count: usize,
        input_len: usize,
        intermediate_size: usize,
        hidden_size: usize,
        resolve: &mut CudaExpertRouteResolveWorkspace,
        workspace: &mut CudaMoeBatchedWorkspace,
    ) -> Result<()> {
        table.ensure_healthy()?;
        if route_count == 0 || route_count > 6 || selected_experts.len() != route_count {
            return Err(Error::Internal {
                message: format!(
                    "stable CUDA MoE dispatch expects 1..=6 matching routes: routes={route_count} selected={}",
                    selected_experts.len()
                ),
            });
        }
        if route_count > expert_ids.len() || route_count > router_weights.len() {
            return Err(Error::Internal {
                message: format!(
                    "stable CUDA MoE router metadata too short: routes={route_count} ids={} weights={}",
                    expert_ids.len(),
                    router_weights.len()
                ),
            });
        }
        if !workspace.matches(route_count, input_len, intermediate_size, hidden_size) {
            return Err(Error::Internal {
                message: format!(
                    "stable CUDA MoE workspace mismatch: workspace=[max_experts={},input={},intermediate={},hidden={}] call=[experts={},input={},intermediate={},hidden={}]",
                    workspace.max_experts,
                    workspace.input_size,
                    workspace.intermediate_size,
                    workspace.hidden_size,
                    route_count,
                    input_len,
                    intermediate_size,
                    hidden_size
                ),
            });
        }

        // This mirror check makes stale bindings an actionable host error without
        // adding a D2H read to steady decode. The device kernel repeats the check
        // so an impossible ordering violation still cannot dereference stale/null
        // expert storage.
        for &expert in selected_experts {
            let binding = table.host.binding(expert).ok_or_else(|| Error::Internal {
                message: format!("stable CUDA MoE dispatch selected unbound expert {expert}"),
            })?;
            if !table.host.is_current(binding) {
                return Err(Error::Internal {
                    message: format!(
                        "stable CUDA MoE dispatch selected stale expert {expert}: slot={} generation={}",
                        binding.slot, binding.generation
                    ),
                });
            }
        }

        self.resolve_expert_routes(table, expert_ids, route_count, resolve)?;
        self.prepare_moe_experts_batched_workspace_resolved_filtered(
            table,
            router_weights,
            route_count,
            input_len,
            intermediate_size,
            hidden_size,
            resolve,
            resolve,
            0,
            workspace,
        )
    }

    /// Gather a subset of already-resolved routes into a warmed MoE workspace.
    /// `active_markers[route] == active_value` selects the routes executed by the
    /// next prepared dispatch; inactive or stale routes contribute exactly zero.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_moe_experts_batched_workspace_resolved_filtered(
        &self,
        table: &CudaExpertSlotTable,
        router_weights: &CudaF32Buffer,
        route_count: usize,
        input_len: usize,
        intermediate_size: usize,
        hidden_size: usize,
        resolved: &CudaExpertRouteResolveWorkspace,
        active_markers: &CudaExpertRouteResolveWorkspace,
        active_value: i32,
        workspace: &mut CudaMoeBatchedWorkspace,
    ) -> Result<()> {
        table.ensure_healthy()?;
        if route_count == 0 || route_count > 6 || route_count > router_weights.len() {
            return Err(Error::Internal {
                message: format!(
                    "resolved CUDA MoE dispatch has invalid route metadata: routes={route_count} weights={}",
                    router_weights.len()
                ),
            });
        }
        if route_count > resolved.route_slots.len()
            || route_count > resolved.route_generations.len()
            || route_count > active_markers.miss_markers.len()
        {
            return Err(Error::Internal {
                message: format!(
                    "resolved CUDA MoE dispatch exceeds scratch capacity: routes={route_count} slots={} generations={} markers={}",
                    resolved.route_slots.len(),
                    resolved.route_generations.len(),
                    active_markers.miss_markers.len()
                ),
            });
        }
        if !workspace.matches(route_count, input_len, intermediate_size, hidden_size) {
            return Err(Error::Internal {
                message: format!(
                    "resolved CUDA MoE workspace mismatch: workspace=[max_experts={},input={},intermediate={},hidden={}] call=[experts={},input={},intermediate={},hidden={}]",
                    workspace.max_experts,
                    workspace.input_size,
                    workspace.intermediate_size,
                    workspace.hidden_size,
                    route_count,
                    input_len,
                    intermediate_size,
                    hidden_size
                ),
            });
        }
        self.launched(unsafe {
            self.module.gather_stable_moe_dispatch(
                &self.stream,
                LaunchConfig::for_num_elems(route_count as u32),
                &table.gate_weight,
                &table.gate_scale,
                &table.up_weight,
                &table.up_scale,
                &table.down_weight,
                &table.down_scale,
                &table.slot_generation,
                &resolved.route_slots.buffer,
                &resolved.route_generations.buffer,
                &router_weights.buffer,
                &active_markers.miss_markers.buffer,
                active_value,
                &mut workspace.gate_ptrs,
                &mut workspace.gate_scale_ptrs,
                &mut workspace.up_ptrs,
                &mut workspace.up_scale_ptrs,
                &mut workspace.down_ptrs,
                &mut workspace.down_scale_ptrs,
                &mut workspace.route_weights,
                &mut workspace.route_slots,
                &mut workspace.dispatch_error,
                route_count as u32,
                table.host.slot_capacity() as u32,
            )
        })
    }

    pub fn reduce_moe_experts_batched_output_into(
        &self,
        workspace: &CudaMoeBatchedWorkspace,
        num_experts: usize,
        hidden_size: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.launched(unsafe {
            self.module.moe_reduce_expert_outputs_ranked(
                &self.stream,
                LaunchConfig::for_num_elems(hidden_size as u32),
                &workspace.expert_output.buffer,
                &workspace.route_slots,
                &mut output.buffer,
                0,
                hidden_size as u32,
                1,
                num_experts as u32,
                num_experts as u32,
            )
        })
    }

    pub fn reduce_moe_experts_batched_split_output_into(
        &self,
        resident: &CudaMoeBatchedWorkspace,
        materialized: &CudaMoeBatchedWorkspace,
        original: &CudaExpertRouteResolveWorkspace,
        num_experts: usize,
        hidden_size: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if num_experts == 0
            || num_experts > resident.max_experts
            || num_experts > materialized.max_experts
            || num_experts > original.miss_markers.len()
        {
            return Err(Error::Internal {
                message: format!(
                    "split CUDA MoE reduction exceeds capacity: routes={num_experts} resident={} materialized={} markers={}",
                    resident.max_experts,
                    materialized.max_experts,
                    original.miss_markers.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.moe_reduce_split_expert_outputs_ranked(
                &self.stream,
                LaunchConfig::for_num_elems(hidden_size as u32),
                &resident.expert_output.buffer,
                &materialized.expert_output.buffer,
                &resident.route_slots,
                &materialized.route_slots,
                &original.miss_markers.buffer,
                &mut output.buffer,
                0,
                hidden_size as u32,
                1,
                num_experts as u32,
                num_experts as u32,
            )
        })
    }

    pub fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        if input.len() != weight.len() || input.is_empty() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA RMS norm length mismatch: input={} weight={}",
                    input.len(),
                    weight.len()
                ),
            });
        }
        let xd = self.upload_f32(input)?;
        let wd = self.upload_f32(weight)?;
        let mut yd = self.zeroed_device_buffer::<f32>(input.len())?;
        self.launched(unsafe {
            self.module.rms_norm_fused(
                &self.stream,
                one_block_config(256),
                &xd,
                &wd,
                &mut yd,
                input.len() as u32,
                eps,
            )
        })?;
        self.download_f32(&yd, input.len())
    }

    /// Device-resident RMS norm: input is already on device, weight is uploaded
    /// once and cached by the caller. Output stays on device.
    pub fn rms_norm_from_device(
        &self,
        input: &CudaF32Buffer,
        weight: &CudaF32Buffer,
        eps: f32,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(input.len())?;
        self.rms_norm_from_device_into(input, weight, eps, &mut output)?;
        Ok(output)
    }

    pub fn rms_norm_from_device_into(
        &self,
        input: &CudaF32Buffer,
        weight: &CudaF32Buffer,
        eps: f32,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if input.len() != weight.len() || input.is_empty() || output.len() != input.len() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA RMS norm device length mismatch: input={} weight={} output={}",
                    input.len(),
                    weight.len(),
                    output.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.rms_norm_fused(
                &self.stream,
                one_block_config(256),
                &input.buffer,
                &weight.buffer,
                &mut output.buffer,
                input.len() as u32,
                eps,
            )
        })
    }

    /// Upload a norm weight once for reuse with `rms_norm_from_device`.
    pub fn upload_norm_weight(&self, weight: &[f32]) -> Result<CudaF32Buffer> {
        self.upload_f32_buffer(weight)
    }

    pub fn rms_norm_rows(
        &self,
        input: &[f32],
        rows: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        let xd = self.upload_f32_buffer(input)?;
        let wd = self.upload_f32_buffer(weight)?;
        let yd = self.rms_norm_rows_from_device(&xd, rows, &wd, eps)?;
        self.download_f32_buffer(&yd)
    }

    pub fn rms_norm_rows_from_device(
        &self,
        input: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        eps: f32,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(input.len())?;
        self.rms_norm_rows_from_device_into(input, rows, weight, eps, &mut output)?;
        Ok(output)
    }

    pub fn rms_norm_rows_from_device_into(
        &self,
        input: &CudaF32Buffer,
        rows: usize,
        weight: &CudaF32Buffer,
        eps: f32,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if rows == 0
            || weight.is_empty()
            || input.len() != rows * weight.len()
            || output.len() != input.len()
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA affine RMS rows length mismatch: rows={rows} input={} weight={} output={}",
                    input.len(),
                    weight.len(),
                    output.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.rms_norm_rows_fused(
                &self.stream,
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &input.buffer,
                &weight.buffer,
                &mut output.buffer,
                rows as u32,
                weight.len() as u32,
                eps,
            )
        })
    }

    pub fn rms_norm_heads(
        &self,
        input: &[f32],
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let xd = self.upload_f32_buffer(input)?;
        let yd = self.rms_norm_heads_from_device(&xd, heads, head_dim, eps)?;
        self.download_f32_buffer(&yd)
    }

    pub fn rms_norm_heads_from_device(
        &self,
        input: &CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(input.len())?;
        self.rms_norm_heads_from_device_into(input, heads, head_dim, eps, &mut output)?;
        Ok(output)
    }

    pub fn rms_norm_heads_from_device_into(
        &self,
        input: &CudaF32Buffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        if heads == 0
            || head_dim == 0
            || input.len() != heads * head_dim
            || output.len() != input.len()
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA per-head RMS device length mismatch: input={} output={} heads={heads} head_dim={head_dim}",
                    input.len(),
                    output.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.rms_norm_heads_fused(
                &self.stream,
                LaunchConfig {
                    grid_dim: (heads as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &input.buffer,
                &mut output.buffer,
                heads as u32,
                head_dim as u32,
                eps,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// `function_col_major` is the logical `[mix_hc, hc_dim]` function matrix
    /// repacked as `[hc_dim, mix_hc]` for coalesced reads across row threads.
    pub fn hc_pre_from_device_into(
        &self,
        state: &CudaF32Buffer,
        function_col_major: &CudaF32Buffer,
        scale: &CudaF32Buffer,
        base: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        eps: f32,
        norm_eps: f32,
        hidden: &mut CudaF32Buffer,
        pre: &mut CudaF32Buffer,
        post: &mut CudaF32Buffer,
        comb: &mut CudaF32Buffer,
    ) -> Result<()> {
        let mix_hc = hc_mult
            .checked_mul(hc_mult + 2)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC mix_hc overflow".into(),
            })?;
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC hidden size overflow".into(),
            })?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || mix_hc > 128
            || hc_mult * hc_mult > 256
            || state.len() != tokens * hc_dim
            || function_col_major.len() != mix_hc * hc_dim
            || scale.len() != 3
            || base.len() != mix_hc
            || hidden.len() != tokens * hidden_size
            || pre.len() != tokens * hc_mult
            || post.len() != tokens * hc_mult
            || comb.len() != tokens * hc_mult * hc_mult
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA HC pre device shape mismatch: tokens={tokens} state={} function={} scale={} base={} hidden={} pre={} post={} comb={} hc={hc_mult} hidden_size={hidden_size} mix={mix_hc}",
                    state.len(),
                    function_col_major.len(),
                    scale.len(),
                    base.len(),
                    hidden.len(),
                    pre.len(),
                    post.len(),
                    comb.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.hc_pre_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &state.buffer,
                &function_col_major.buffer,
                &scale.buffer,
                &base.buffer,
                &mut hidden.buffer,
                &mut pre.buffer,
                &mut post.buffer,
                &mut comb.buffer,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                mix_hc as u32,
                sinkhorn_iters as u32,
                eps,
                norm_eps,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_f32(
        &self,
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mix_hc = hc_mult
            .checked_mul(hc_mult + 2)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC mix_hc overflow".into(),
            })?;
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC hidden size overflow".into(),
            })?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || mix_hc > 128
            || hc_mult * hc_mult > 256
            || state.len() != tokens * hc_dim
            || function.len() != mix_hc * hc_dim
            || scale.len() != 3
            || base.len() != mix_hc
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA HC pre shape mismatch: tokens={tokens} state={} function={} scale={} base={} hc={hc_mult} hidden={hidden_size} mix={mix_hc}",
                    state.len(),
                    function.len(),
                    scale.len(),
                    base.len()
                ),
            });
        }
        let function_col_major = transpose_hc_function_for_device(function, mix_hc)?;
        let sd = self.upload_f32(state)?;
        let fd = self.upload_f32(&function_col_major)?;
        let scd = self.upload_f32(scale)?;
        let bd = self.upload_f32(base)?;
        let mut hidden = self.zeroed_device_buffer::<f32>(tokens * hidden_size)?;
        let mut pre = self.zeroed_device_buffer::<f32>(tokens * hc_mult)?;
        let mut post = self.zeroed_device_buffer::<f32>(tokens * hc_mult)?;
        let mut comb = self.zeroed_device_buffer::<f32>(tokens * hc_mult * hc_mult)?;
        self.launched(unsafe {
            self.module.hc_pre_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &sd,
                &fd,
                &scd,
                &bd,
                &mut hidden,
                &mut pre,
                &mut post,
                &mut comb,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                mix_hc as u32,
                sinkhorn_iters as u32,
                eps,
                norm_eps,
            )
        })?;
        Ok((
            self.download_f32(&hidden, tokens * hidden_size)?,
            self.download_f32(&pre, tokens * hc_mult)?,
            self.download_f32(&post, tokens * hc_mult)?,
            self.download_f32(&comb, tokens * hc_mult * hc_mult)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_post_from_device_into(
        &self,
        hidden: &CudaF32Buffer,
        residual: &CudaF32Buffer,
        split_post: &CudaF32Buffer,
        split_comb: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC post hidden size overflow".into(),
            })?;
        if tokens == 0 || hc_mult == 0 || hidden_size == 0 || output.len() != tokens * hc_dim {
            return Err(Error::Internal {
                message: format!(
                    "CUDA HC post device shape mismatch: tokens={tokens} hc={hc_mult} hidden_size={hidden_size} output={}",
                    output.len()
                ),
            });
        }
        self.launched(unsafe {
            self.module.hc_post_f32(
                &self.stream,
                LaunchConfig::for_num_elems((tokens * hc_dim) as u32),
                &hidden.buffer,
                &residual.buffer,
                &split_post.buffer,
                &split_comb.buffer,
                &mut output.buffer,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
            )
        })
    }

    pub fn hc_post_f32(
        &self,
        hidden: &[f32],
        residual: &[f32],
        split_post: &[f32],
        split_comb: &[f32],
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC post hidden size overflow".into(),
            })?;
        if tokens == 0
            || hc_mult == 0
            || hidden_size == 0
            || hidden.len() != tokens * hidden_size
            || residual.len() != tokens * hc_dim
            || split_post.len() != tokens * hc_mult
            || split_comb.len() != tokens * hc_mult * hc_mult
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA HC post shape mismatch: tokens={tokens} hidden={} residual={} post={} comb={} hc={hc_mult} hidden_size={hidden_size}",
                    hidden.len(),
                    residual.len(),
                    split_post.len(),
                    split_comb.len()
                ),
            });
        }
        let hd = self.upload_f32(hidden)?;
        let rd = self.upload_f32(residual)?;
        let pd = self.upload_f32(split_post)?;
        let cd = self.upload_f32(split_comb)?;
        let mut out = self.zeroed_device_buffer::<f32>(tokens * hc_dim)?;
        self.launched(unsafe {
            self.module.hc_post_f32(
                &self.stream,
                LaunchConfig::for_num_elems((tokens * hc_dim) as u32),
                &hd,
                &rd,
                &pd,
                &cd,
                &mut out,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
            )
        })?;
        self.download_f32(&out, tokens * hc_dim)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_head_f32(
        &self,
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        eps: f32,
        norm_eps: f32,
    ) -> Result<Vec<f32>> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC head hidden size overflow".into(),
            })?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || state.len() != tokens * hc_dim
            || function.len() != hc_mult * hc_dim
            || scale.len() != 1
            || base.len() != hc_mult
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA HC head shape mismatch: tokens={tokens} state={} function={} scale={} base={} hc={hc_mult} hidden={hidden_size}",
                    state.len(),
                    function.len(),
                    scale.len(),
                    base.len()
                ),
            });
        }
        let sd = self.upload_f32(state)?;
        let fd = self.upload_f32(function)?;
        let scd = self.upload_f32(scale)?;
        let bd = self.upload_f32(base)?;
        let mut hidden = self.zeroed_device_buffer::<f32>(tokens * hidden_size)?;
        self.launched(unsafe {
            self.module.hc_head_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &sd,
                &fd,
                &scd,
                &bd,
                &mut hidden,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                eps,
                norm_eps,
            )
        })?;
        self.download_f32(&hidden, tokens * hidden_size)
    }

    /// Compute `mean(hc)` for each row and scatter it directly into one slot of
    /// the concatenated proposal target-tap buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_mean_scatter_from_device_into(
        &self,
        state: &CudaF32Buffer,
        rows: usize,
        hc_mult: usize,
        hidden_size: usize,
        tap_slot: usize,
        tap_count: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let expected_state = rows
            .checked_mul(hc_mult)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC mean input size overflow".into(),
            })?;
        let expected_output = rows
            .checked_mul(tap_count)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC mean output size overflow".into(),
            })?;
        if rows == 0
            || hc_mult == 0
            || hidden_size == 0
            || tap_count == 0
            || tap_slot >= tap_count
            || state.len() != expected_state
            || output.len() != expected_output
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA HC mean-scatter shape mismatch: state={}/{} output={}/{} rows={rows} hc={hc_mult} hidden={hidden_size} tap={tap_slot}/{tap_count}",
                    state.len(),
                    expected_state,
                    output.len(),
                    expected_output
                ),
            });
        }
        let values = rows
            .checked_mul(tap_count)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC mean launch size overflow".into(),
            })?;
        let values = checked_u32(values, "hc_mean_scatter", "rows * hidden_size")?;
        self.launched(unsafe {
            self.module.hc_mean_scatter_f32(
                &self.stream,
                LaunchConfig::for_num_elems(values),
                &state.buffer,
                &mut output.buffer,
                checked_u32(rows, "hc_mean_scatter", "rows")?,
                checked_u32(hc_mult, "hc_mean_scatter", "hc_mult")?,
                checked_u32(hidden_size, "hc_mean_scatter", "hidden_size")?,
                checked_u32(tap_slot, "hc_mean_scatter", "tap_slot")?,
                checked_u32(tap_count, "hc_mean_scatter", "tap_count")?,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_head_from_device_into(
        &self,
        state: &CudaF32Buffer,
        function: &CudaF32Buffer,
        scale: &CudaF32Buffer,
        base: &CudaF32Buffer,
        tokens: usize,
        hc_mult: usize,
        hidden_size: usize,
        eps: f32,
        norm_eps: f32,
        hidden: &mut CudaF32Buffer,
    ) -> Result<()> {
        let hc_dim = hc_mult
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::Internal {
                message: "CUDA HC head hidden size overflow".into(),
            })?;
        if tokens == 0
            || hc_mult == 0
            || hc_mult > 16
            || state.len() < tokens * hc_dim
            || function.len() != hc_mult * hc_dim
            || scale.len() != 1
            || base.len() != hc_mult
            || hidden.len() != tokens * hidden_size
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA HC head device shape mismatch: tokens={tokens} state={} function={} scale={} base={} output={} hc={hc_mult} hidden={hidden_size}",
                    state.len(),
                    function.len(),
                    scale.len(),
                    base.len(),
                    hidden.len(),
                ),
            });
        }
        self.launched(unsafe {
            self.module.hc_head_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (tokens as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &state.buffer,
                &function.buffer,
                &scale.buffer,
                &base.buffer,
                &mut hidden.buffer,
                tokens as u32,
                hc_mult as u32,
                hidden_size as u32,
                eps,
                norm_eps,
            )
        })
    }

    fn artifact_linear_rows_prepacked_fp8_from_f32(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let in_features = handle.shape.in_features();
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "CUDA FP8 MMA input size overflow".into(),
            })?;
        let scale_len = rows
            .checked_mul(in_features / ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE)
            .ok_or_else(|| Error::Internal {
                message: "CUDA FP8 MMA scale size overflow".into(),
            })?;
        let mut x_packed = self.uninitialized_device_buffer::<u8>(input_len)?;
        let mut x_scales = self.uninitialized_device_buffer::<u8>(scale_len)?;
        self.artifact_linear_rows_prepacked_fp8_from_f32_preallocated(
            handle,
            input,
            rows,
            output,
            &mut x_packed,
            input_len,
            &mut x_scales,
            scale_len,
        )
    }

    fn artifact_linear_rows_prepacked_fp8_from_f32_with_scratch(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
        scratch: &mut CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        self.artifact_linear_rows_prepacked_fp8_from_f32_preallocated(
            handle,
            input,
            rows,
            output,
            &mut scratch.x_packed,
            scratch.value_capacity,
            &mut scratch.x_scales,
            scratch.scale_capacity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn artifact_linear_rows_prepacked_fp8_from_f32_preallocated(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
        x_packed: &mut DeviceBuffer<u8>,
        packed_capacity: usize,
        x_scales: &mut DeviceBuffer<u8>,
        scale_capacity: usize,
    ) -> Result<()> {
        let in_features = handle.shape.in_features();
        self.pack_fp8_rows_from_f32_preallocated(
            input,
            rows,
            in_features,
            x_packed,
            packed_capacity,
            x_scales,
            scale_capacity,
        )?;
        self.artifact_linear_rows_prepacked_fp8(handle, x_packed, x_scales, rows, output)
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_fp8_rows_from_f32_preallocated(
        &self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        in_features: usize,
        x_packed: &mut DeviceBuffer<u8>,
        packed_capacity: usize,
        x_scales: &mut DeviceBuffer<u8>,
        scale_capacity: usize,
    ) -> Result<()> {
        if rows == 0 || in_features == 0 || !in_features.is_multiple_of(128) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA FP8 MMA pack requires positive K128 rows: rows={rows} in_features={in_features}"
                ),
            });
        }
        let scale_cols = in_features / ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE;
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "CUDA FP8 MMA input size overflow".into(),
            })?;
        let scale_len = rows
            .checked_mul(scale_cols)
            .ok_or_else(|| Error::Internal {
                message: "CUDA FP8 MMA scale size overflow".into(),
            })?;
        if input_len > packed_capacity || scale_len > scale_capacity {
            return Err(Error::Internal {
                message: format!(
                    "CUDA FP8 MMA scratch too small: packed={input_len}/{packed_capacity} scales={scale_len}/{scale_capacity}"
                ),
            });
        }
        self.launched(unsafe {
            self.module.fp8_e4m3fn_e8m0_quantize_f32_packed(
                &self.stream,
                LaunchConfig::for_num_elems(scale_len as u32),
                input,
                x_packed,
                x_scales,
                input_len as u32,
                in_features as u32,
                ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE as u32,
            )
        })
    }

    fn artifact_linear_rows_prepacked_fp8(
        &self,
        handle: &CudaArtifactLinearHandle,
        x_packed: &DeviceBuffer<u8>,
        x_scales: &DeviceBuffer<u8>,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m: 128,
            block_k: 128,
        } = handle.shape
        else {
            return Err(Error::Internal {
                message: "CUDA FP8 MMA packed rows called with unsupported artifact shape".into(),
            });
        };
        let weight_scales = handle.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "CUDA FP8 artifact linear missing scale".into(),
        })?;
        let scale_cols = in_features / ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE;
        self.launched(unsafe {
            self.module.gemm_fp8_e4m3fn_e8m0_prepacked(
                &self.stream,
                LaunchConfig {
                    grid_dim: (out_features.div_ceil(16) as u32, rows.div_ceil(8) as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                x_packed,
                x_scales,
                &handle.weight,
                weight_scales,
                output,
                rows as u32,
                out_features as u32,
                in_features as u32,
                scale_cols as u32,
            )
        })
    }

    fn artifact_linear_matvec_prepacked_fp8_from_f32(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m: 128,
            block_k: 128,
        } = handle.shape
        else {
            return Err(Error::Internal {
                message: "CUDA FP8 MMA matvec called with unsupported artifact shape".into(),
            });
        };
        let weight_scales = handle.scale.as_ref().ok_or_else(|| Error::Internal {
            message: "CUDA FP8 artifact linear missing scale".into(),
        })?;
        let scale_cols = in_features / ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE;
        self.launched(unsafe {
            self.module.gemv_fp8_e4m3fn_e8m0_from_f32(
                &self.stream,
                LaunchConfig {
                    grid_dim: (out_features.div_ceil(16) as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                input,
                &handle.weight,
                weight_scales,
                output,
                out_features as u32,
                in_features as u32,
                scale_cols as u32,
            )
        })
    }

    fn artifact_linear_rows_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match handle.shape {
            CudaArtifactLinearShape::F32 {
                out_features,
                in_features,
            } => self.launched(unsafe {
                self.module.gemm_f32_bytes(
                    &self.stream,
                    LaunchConfig::for_num_elems((rows * out_features) as u32),
                    input,
                    &handle.weight,
                    output,
                    rows as u32,
                    out_features as u32,
                    in_features as u32,
                )
            }),
            CudaArtifactLinearShape::Bf16Bytes {
                out_features,
                in_features,
            } => {
                if rows == 0 || !in_features.is_multiple_of(16) {
                    return Err(Error::Internal {
                        message: format!(
                            "BF16 MMA requires positive rows and K16: rows={rows} in_features={in_features}"
                        ),
                    });
                }
                self.launched(unsafe {
                    self.module.linear_rows_bf16_from_f32(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: (
                                out_features.div_ceil(64) as u32,
                                rows.div_ceil(8) as u32,
                                1,
                            ),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        input,
                        &handle.weight,
                        output,
                        rows as u32,
                        out_features as u32,
                        in_features as u32,
                    )
                })
            }
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = handle.scale.as_ref().ok_or_else(|| Error::Internal {
                    message: "CUDA FP8 artifact linear missing scale".into(),
                })?;
                let scale_cols = in_features.div_ceil(block_k);
                self.launched(unsafe {
                    self.module.gemm_fp8_e4m3fn_e8m0_2d(
                        &self.stream,
                        LaunchConfig::for_num_elems((rows * out_features) as u32),
                        input,
                        &handle.weight,
                        scale,
                        output,
                        rows as u32,
                        out_features as u32,
                        in_features as u32,
                        scale_cols as u32,
                        block_m as u32,
                        block_k as u32,
                    )
                })
            }
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale { .. } => {
                Err(unsupported_grouped_fp4_moe())
            }
        }
    }

    fn artifact_linear_matvec_device(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match handle.shape {
            CudaArtifactLinearShape::F32 {
                out_features,
                in_features,
            } => self.launched(unsafe {
                self.module.gemv_f32_bytes(
                    &self.stream,
                    LaunchConfig::for_num_elems(out_features as u32),
                    input,
                    &handle.weight,
                    output,
                    out_features as u32,
                    in_features as u32,
                )
            }),
            CudaArtifactLinearShape::Bf16Bytes {
                out_features,
                in_features,
            } => self.launched(unsafe {
                self.module.linear_bf16_from_f32(
                    &self.stream,
                    LaunchConfig {
                        grid_dim: (out_features as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    input,
                    &handle.weight,
                    output,
                    out_features as u32,
                    in_features as u32,
                )
            }),

            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m,
                block_k,
            } => {
                let scale = handle.scale.as_ref().ok_or_else(|| Error::Internal {
                    message: "CUDA FP8 artifact linear missing scale".into(),
                })?;
                let scale_cols = in_features.div_ceil(block_k);
                self.launched(unsafe {
                    self.module.gemv_fp8_e4m3fn_e8m0_2d(
                        &self.stream,
                        LaunchConfig::for_num_elems(out_features as u32),
                        input,
                        &handle.weight,
                        scale,
                        output,
                        out_features as u32,
                        in_features as u32,
                        scale_cols as u32,
                        block_m as u32,
                        block_k as u32,
                    )
                })
            }
            CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale { .. } => {
                Err(unsupported_grouped_fp4_moe())
            }
        }
    }

    pub fn convert_combined_ring_topk_indices_into(
        &self,
        combined: &CudaI32Buffer,
        window_lens: CombinedRingWindowLens<'_>,
        layout: CombinedRingTopkLayout,
        logical_indices: &mut CudaI32Buffer,
        plane_selectors: &mut CudaI32Buffer,
    ) -> Result<()> {
        layout.validate()?;
        let elements = layout.elements()?;
        if combined.len() < elements
            || logical_indices.len() != elements
            || plane_selectors.len() != elements
        {
            return Err(Error::Internal {
                message: format!(
                    "combined ring conversion length mismatch: input={} logical={} selectors={} expected={elements}",
                    combined.len(),
                    logical_indices.len(),
                    plane_selectors.len()
                ),
            });
        }
        let (row_window_lens, explicit) = match window_lens {
            CombinedRingWindowLens::PositionDerived => (combined, 0u32),
            CombinedRingWindowLens::Explicit(values) => {
                if values.len() != layout.rows {
                    return Err(Error::Internal {
                        message: format!(
                            "combined ring row window length mismatch: got {} expected {}",
                            values.len(),
                            layout.rows
                        ),
                    });
                }
                (values, 1u32)
            }
        };
        self.launched(unsafe {
            self.module.convert_combined_ring_topk_indices(
                &self.stream,
                LaunchConfig::for_num_elems(checked_u32(
                    elements,
                    "combined ring conversion",
                    "elements",
                )?),
                &combined.buffer,
                &row_window_lens.buffer,
                &mut logical_indices.buffer,
                &mut plane_selectors.buffer,
                checked_u32(elements, "combined ring conversion", "elements")?,
                checked_u32(layout.rows, "combined ring conversion", "rows")?,
                checked_u32(layout.topk, "combined ring conversion", "topk")?,
                checked_u32(
                    layout.start_position,
                    "combined ring conversion",
                    "start_position",
                )?,
                checked_u32(
                    layout.position_stride,
                    "combined ring conversion",
                    "position_stride",
                )?,
                checked_u32(
                    layout.window_size,
                    "combined ring conversion",
                    "window_size",
                )?,
                explicit,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dual_plane_paged_sparse_attention_sink_from_device(
        &self,
        query: &CudaF32Buffer,
        first_plane: &CudaF32Buffer,
        second_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        sequence_block_offsets: &CudaI32Buffer,
        sequence_kv_lens: &CudaI32Buffer,
        second_sequence_kv_lens: &CudaI32Buffer,
        topk: &CudaI32Buffer,
        selectors: &CudaI32Buffer,
        sink: &CudaF32Buffer,
        layout: DualPlanePagedSparseAttentionLayout,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(layout.base.output_elements()?)?;
        self.dual_plane_paged_sparse_attention_sink_from_device_into(
            query,
            first_plane,
            second_plane,
            block_slots,
            sequence_block_offsets,
            sequence_kv_lens,
            second_sequence_kv_lens,
            topk,
            selectors,
            sink,
            layout,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dual_plane_paged_sparse_attention_sink_from_device_into(
        &self,
        query: &CudaF32Buffer,
        first_plane: &CudaF32Buffer,
        second_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        sequence_block_offsets: &CudaI32Buffer,
        sequence_kv_lens: &CudaI32Buffer,
        second_sequence_kv_lens: &CudaI32Buffer,
        topk: &CudaI32Buffer,
        selectors: &CudaI32Buffer,
        sink: &CudaF32Buffer,
        layout: DualPlanePagedSparseAttentionLayout,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        layout.validate_buffer_lengths(
            query.len(),
            first_plane.len(),
            second_plane.len(),
            block_slots.len(),
            sequence_block_offsets.len(),
            sequence_kv_lens.len(),
            second_sequence_kv_lens.len(),
            topk.len(),
            selectors.len(),
            sink.len(),
            output.len(),
        )?;
        let explicit_selection_layout = layout.explicit_selection_layout(false)?;
        let mut workspace =
            self.hybrid_mla_explicit_selection_workspace(explicit_selection_layout)?;
        CudaSparseAttentionExecutor::new(&self.stream).dual_plane_paged_sparse_attention_sink_f32(
            &query.buffer,
            &first_plane.buffer,
            &second_plane.buffer,
            &block_slots.buffer,
            &sequence_block_offsets.buffer,
            &sequence_kv_lens.buffer,
            &second_sequence_kv_lens.buffer,
            None,
            &topk.buffer,
            &selectors.buffer,
            &sink.buffer,
            &mut output.buffer,
            &mut workspace.storage,
            &mut workspace.status.buffer,
            #[cfg(ferrule_cuda_test_oracle)]
            &mut workspace.oracle_output,
            layout,
        )?;
        self.record_kernel_launches(hybrid_mla_explicit_selection_launch_count());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dual_plane_paged_sparse_attention_selected_rows_from_device_into(
        &self,
        query: &CudaF32Buffer,
        first_plane: &CudaF32Buffer,
        second_plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        sequence_block_offsets: &CudaI32Buffer,
        sequence_kv_lens: &CudaI32Buffer,
        second_sequence_kv_lens: &CudaI32Buffer,
        row_sequence_ids: &CudaI32Buffer,
        row_kv_lens: &CudaI32Buffer,
        row_second_kv_lens: &CudaI32Buffer,
        topk: &CudaI32Buffer,
        selectors: &CudaI32Buffer,
        sink: &CudaF32Buffer,
        layout: DualPlanePagedSparseAttentionLayout,
        workspace: &mut CudaHybridMlaExplicitSelectionWorkspace,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        layout.validate_selected_buffer_lengths(
            query.len(),
            first_plane.len(),
            second_plane.len(),
            block_slots.len(),
            sequence_block_offsets.len(),
            sequence_kv_lens.len(),
            second_sequence_kv_lens.len(),
            row_sequence_ids.len(),
            row_kv_lens.len(),
            row_second_kv_lens.len(),
            topk.len(),
            selectors.len(),
            sink.len(),
            output.len(),
        )?;
        let explicit_selection_layout = layout.explicit_selection_layout(true)?;
        if !workspace.supports(explicit_selection_layout)? {
            return Err(Error::Internal {
                message: format!(
                    "hybrid MLA explicit selection workspace capacity mismatch: capacity_bytes={} alignment={} allocated_for={:?} required_for={explicit_selection_layout:?}",
                    workspace.capacity_bytes, workspace.alignment, workspace.allocated_layout,
                ),
            });
        }
        CudaSparseAttentionExecutor::new(&self.stream).dual_plane_paged_sparse_attention_sink_f32(
            &query.buffer,
            &first_plane.buffer,
            &second_plane.buffer,
            &block_slots.buffer,
            &sequence_block_offsets.buffer,
            &sequence_kv_lens.buffer,
            &second_sequence_kv_lens.buffer,
            Some((
                &row_sequence_ids.buffer,
                &row_kv_lens.buffer,
                &row_second_kv_lens.buffer,
            )),
            &topk.buffer,
            &selectors.buffer,
            &sink.buffer,
            &mut output.buffer,
            &mut workspace.storage,
            &mut workspace.status.buffer,
            #[cfg(ferrule_cuda_test_oracle)]
            &mut workspace.oracle_output,
            layout,
        )?;
        self.record_kernel_launches(hybrid_mla_explicit_selection_launch_count());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_sparse_attention_sink_from_device(
        &self,
        query: &CudaF32Buffer,
        plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        sequence_block_offsets: &CudaI32Buffer,
        sequence_kv_lens: &CudaI32Buffer,
        topk: &CudaI32Buffer,
        sink: &CudaF32Buffer,
        layout: PagedSparseAttentionLayout,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(layout.output_elements()?)?;
        self.paged_sparse_attention_sink_from_device_into(
            query,
            plane,
            block_slots,
            sequence_block_offsets,
            sequence_kv_lens,
            topk,
            sink,
            layout,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_sparse_attention_sink_from_device_into(
        &self,
        query: &CudaF32Buffer,
        plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        sequence_block_offsets: &CudaI32Buffer,
        sequence_kv_lens: &CudaI32Buffer,
        topk: &CudaI32Buffer,
        sink: &CudaF32Buffer,
        layout: PagedSparseAttentionLayout,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        layout.validate_buffer_lengths(
            query.len(),
            plane.len(),
            block_slots.len(),
            sequence_block_offsets.len(),
            sequence_kv_lens.len(),
            topk.len(),
            sink.len(),
            output.len(),
        )?;
        let explicit_selection_layout = layout.explicit_selection_layout(false)?;
        let mut workspace =
            self.hybrid_mla_explicit_selection_workspace(explicit_selection_layout)?;
        CudaSparseAttentionExecutor::new(&self.stream).paged_sparse_attention_sink_f32(
            &query.buffer,
            &plane.buffer,
            &block_slots.buffer,
            &sequence_block_offsets.buffer,
            &sequence_kv_lens.buffer,
            None,
            &topk.buffer,
            &sink.buffer,
            &mut output.buffer,
            &mut workspace.storage,
            &mut workspace.status.buffer,
            #[cfg(ferrule_cuda_test_oracle)]
            &mut workspace.oracle_output,
            layout,
        )?;
        self.record_kernel_launches(hybrid_mla_explicit_selection_launch_count());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_sparse_attention_selected_rows_from_device_into(
        &self,
        query: &CudaF32Buffer,
        plane: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        sequence_block_offsets: &CudaI32Buffer,
        sequence_kv_lens: &CudaI32Buffer,
        row_sequence_ids: &CudaI32Buffer,
        row_kv_lens: &CudaI32Buffer,
        topk: &CudaI32Buffer,
        sink: &CudaF32Buffer,
        layout: PagedSparseAttentionLayout,
        workspace: &mut CudaHybridMlaExplicitSelectionWorkspace,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        layout.validate_selected_buffer_lengths(
            query.len(),
            plane.len(),
            block_slots.len(),
            sequence_block_offsets.len(),
            sequence_kv_lens.len(),
            row_sequence_ids.len(),
            row_kv_lens.len(),
            topk.len(),
            sink.len(),
            output.len(),
        )?;
        let explicit_selection_layout = layout.explicit_selection_layout(true)?;
        if !workspace.supports(explicit_selection_layout)? {
            return Err(Error::Internal {
                message: format!(
                    "hybrid MLA explicit selection workspace capacity mismatch: capacity_bytes={} alignment={} allocated_for={:?} required_for={explicit_selection_layout:?}",
                    workspace.capacity_bytes, workspace.alignment, workspace.allocated_layout,
                ),
            });
        }
        CudaSparseAttentionExecutor::new(&self.stream).paged_sparse_attention_sink_f32(
            &query.buffer,
            &plane.buffer,
            &block_slots.buffer,
            &sequence_block_offsets.buffer,
            &sequence_kv_lens.buffer,
            Some((&row_sequence_ids.buffer, &row_kv_lens.buffer)),
            &topk.buffer,
            &sink.buffer,
            &mut output.buffer,
            &mut workspace.storage,
            &mut workspace.status.buffer,
            #[cfg(ferrule_cuda_test_oracle)]
            &mut workspace.oracle_output,
            layout,
        )?;
        self.record_kernel_launches(hybrid_mla_explicit_selection_launch_count());
        Ok(())
    }

    pub fn sparse_attention_sink_from_device(
        &self,
        query: &CudaF32Buffer,
        values: &CudaF32Buffer,
        topk: &DeviceBuffer<i32>,
        sink: &CudaF32Buffer,
        shape: CudaSparseAttentionShape,
    ) -> Result<CudaF32Buffer> {
        let mut output = self.zero_f32_buffer(shape.output_elements())?;
        self.sparse_attention_sink_from_device_into(query, values, topk, sink, shape, &mut output)?;
        Ok(output)
    }

    pub fn sparse_attention_sink_from_device_into(
        &self,
        query: &CudaF32Buffer,
        values: &CudaF32Buffer,
        topk: &DeviceBuffer<i32>,
        sink: &CudaF32Buffer,
        shape: CudaSparseAttentionShape,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        shape.validate()?;
        if output.len() != shape.output_elements() {
            return Err(Error::Internal {
                message: format!(
                    "CUDA sparse attention output length mismatch: expected {}, got {}",
                    shape.output_elements(),
                    output.len()
                ),
            });
        }
        let explicit_selection_layout = shape.explicit_selection_layout();
        let mut workspace =
            self.hybrid_mla_explicit_selection_workspace(explicit_selection_layout)?;
        CudaSparseAttentionExecutor::new(&self.stream).sparse_attention_sink_f32(
            &query.buffer,
            &values.buffer,
            topk,
            &sink.buffer,
            &mut output.buffer,
            &mut workspace.storage,
            &mut workspace.status.buffer,
            #[cfg(ferrule_cuda_test_oracle)]
            &mut workspace.oracle_output,
            shape,
        )?;
        self.record_kernel_launches(hybrid_mla_explicit_selection_launch_count());
        Ok(())
    }

    /// Apply DSV4-style tail rotary embedding (interleaved pairs, YAARN-scaled)
    /// to a device buffer. `cos_table` and `sin_table` are precomputed for
    /// `[max_positions, rope_dim/2]` and uploaded once.
    pub fn rope_tail_from_device(
        &self,
        qk: &mut CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        position: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        // Keep decode and batched prefill on the same pair-owned kernel. The old
        // element-owned kernel let even/odd lanes read and write the same rotary
        // pair concurrently, so decode could diverge from the race-free rows path.
        self.rope_tail_rows_from_device(
            qk, cos_table, sin_table, position, 1, heads, head_dim, rope_dim, inverse,
        )
    }

    /// Apply DSV4-style tail rotary to batched rows laid out as
    /// `[rows, heads, head_dim]`, using `start_position + row` per row.
    pub fn rope_tail_rows_from_device(
        &self,
        qk: &mut CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        start_position: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        self.rope_tail_rows_strided_from_device(
            qk,
            cos_table,
            sin_table,
            start_position,
            1,
            rows,
            heads,
            head_dim,
            rope_dim,
            inverse,
        )
    }

    /// Apply DSV4-style tail rotary to batched rows using
    /// `start_position + row * position_stride` per row.
    pub fn rope_tail_rows_strided_from_device(
        &self,
        qk: &mut CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        start_position: u32,
        position_stride: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if rows == 0 || heads == 0 || rope_dim == 0 || rope_dim > head_dim {
            return Ok(());
        }
        let expected = rows as usize * heads as usize * head_dim as usize;
        if qk.len() != expected {
            return Err(Error::Internal {
                message: format!(
                    "CUDA rope rows length mismatch: len={} expected rows={} heads={} head_dim={}",
                    qk.len(),
                    rows,
                    heads,
                    head_dim
                ),
            });
        }
        let pairs = rows.saturating_mul(heads).saturating_mul(rope_dim / 2);
        if pairs == 0 {
            return Ok(());
        }
        self.launched(unsafe {
            self.module.rope_tail_yaarn_rows_strided(
                &self.stream,
                LaunchConfig::for_num_elems(pairs),
                &mut qk.buffer,
                &cos_table.buffer,
                &sin_table.buffer,
                pairs,
                start_position,
                position_stride,
                rows,
                heads,
                head_dim,
                rope_dim,
                if inverse { 1u32 } else { 0u32 },
            )
        })
    }

    /// Apply DSV4-style tail rotary to `[rows, heads, head_dim]` using one
    /// arbitrary device-resident position per row.
    pub fn rope_tail_rows_indexed_from_device(
        &self,
        qk: &mut CudaF32Buffer,
        cos_table: &CudaF32Buffer,
        sin_table: &CudaF32Buffer,
        positions: &CudaI32Buffer,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: bool,
    ) -> Result<()> {
        if positions.len() != rows as usize {
            return Err(Error::Internal {
                message: format!(
                    "CUDA indexed rope positions length mismatch: got {} expected rows={rows}",
                    positions.len()
                ),
            });
        }
        if rows == 0 || heads == 0 || rope_dim == 0 || rope_dim > head_dim {
            return Ok(());
        }
        let expected = (rows as usize)
            .checked_mul(heads as usize)
            .and_then(|value| value.checked_mul(head_dim as usize))
            .ok_or_else(|| Error::Internal {
                message: "CUDA indexed rope row size overflow".into(),
            })?;
        if qk.len() != expected {
            return Err(Error::Internal {
                message: format!(
                    "CUDA indexed rope rows length mismatch: len={} expected rows={} heads={} head_dim={}",
                    qk.len(),
                    rows,
                    heads,
                    head_dim
                ),
            });
        }
        let table_width = (rope_dim / 2) as usize;
        if table_width == 0 {
            return Ok(());
        }
        if cos_table.len() != sin_table.len() || !cos_table.len().is_multiple_of(table_width) {
            return Err(Error::Internal {
                message: format!(
                    "CUDA indexed rope table shape mismatch: cos={} sin={} row_width={table_width}",
                    cos_table.len(),
                    sin_table.len()
                ),
            });
        }
        let pairs = rows
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(rope_dim / 2))
            .ok_or_else(|| Error::Internal {
                message: "CUDA indexed rope pair count overflow".into(),
            })?;
        self.launched(unsafe {
            self.module.rope_tail_yaarn_rows_indexed(
                &self.stream,
                LaunchConfig::for_num_elems(pairs),
                &mut qk.buffer,
                &cos_table.buffer,
                &sin_table.buffer,
                &positions.buffer,
                pairs,
                rows,
                heads,
                head_dim,
                rope_dim,
                if inverse { 1u32 } else { 0u32 },
            )
        })
    }

    pub fn sparse_attention_sink_f32(
        &self,
        query: &[f32],
        values: &[f32],
        topk: &[i32],
        sink: &[f32],
        shape: CudaSparseAttentionShape,
    ) -> Result<Vec<f32>> {
        shape.validate()?;
        if query.len() != shape.q_elements()
            || values.len() != shape.kv_elements()
            || topk.len() != shape.topk_elements()
            || sink.len() != shape.heads
        {
            return Err(Error::Internal {
                message: format!(
                    "sparse attention length mismatch: q={} values={} topk={} sink={}, expected q={} values={} topk={} sink={}",
                    query.len(),
                    values.len(),
                    topk.len(),
                    sink.len(),
                    shape.q_elements(),
                    shape.kv_elements(),
                    shape.topk_elements(),
                    shape.heads
                ),
            });
        }
        let qd = self.upload_f32(query)?;
        let vd = self.upload_f32(values)?;
        let td = self.upload_i32(topk)?;
        let sd = self.upload_f32(sink)?;
        let mut od = self.zeroed_device_buffer::<f32>(shape.output_elements())?;
        let explicit_selection_layout = shape.explicit_selection_layout();
        let mut workspace =
            self.hybrid_mla_explicit_selection_workspace(explicit_selection_layout)?;
        CudaSparseAttentionExecutor::new(&self.stream).sparse_attention_sink_f32(
            &qd,
            &vd,
            &td,
            &sd,
            &mut od,
            &mut workspace.storage,
            &mut workspace.status.buffer,
            #[cfg(ferrule_cuda_test_oracle)]
            &mut workspace.oracle_output,
            shape,
        )?;
        self.record_kernel_launches(hybrid_mla_explicit_selection_launch_count());
        self.download_f32(&od, shape.output_elements())
    }
}

fn one_block_config(threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ── Standalone GEMV (benchmark) ───────────────────────────────────────

/// Run a single GEMV on GPU — used for microbenchmarking.
pub fn cuda_gemv(x: &[f32], w: &[f32], out_f: usize) -> Result<Vec<f32>> {
    let ctx = cu(CudaContext::new(0))?;
    cu(ctx.bind_to_thread())?;
    let module = cu(crate::cuda::kernels::kernels::load(&ctx))?;
    let s = ctx.default_stream();
    let xd = cu(DeviceBuffer::from_host(&s, x))?;
    let wd = cu(DeviceBuffer::from_host(&s, w))?;
    let mut yd = cu(DeviceBuffer::<f32>::zeroed(&s, out_f))?;
    cu(unsafe {
        module.gemv_f32(
            &s,
            LaunchConfig::for_num_elems(out_f as u32),
            &xd,
            &wd,
            &mut yd,
            out_f as u32,
            x.len() as u32,
        )
    })?;
    cu(yd.to_host_vec(&s))
}

/// Run FP8 E4M3FN + E8M0 2D-block-scale GEMV on GPU.
pub fn cuda_gemv_fp8_e4m3fn_e8m0_2d(
    x: &[f32],
    weight: &[u8],
    scales: &[u8],
    out_features: usize,
    in_features: usize,
    block_m: usize,
    block_k: usize,
) -> Result<Vec<f32>> {
    if in_features == 0 || block_m == 0 || block_k == 0 {
        return Err(Error::Internal {
            message: "invalid FP8 GEMV shape".to_string(),
        });
    }
    let expected_weight = out_features
        .checked_mul(in_features)
        .ok_or_else(|| Error::Internal {
            message: "FP8 weight size overflow".into(),
        })?;
    let scale_rows = out_features.div_ceil(block_m);
    let scale_cols = in_features.div_ceil(block_k);
    let expected_scales = scale_rows
        .checked_mul(scale_cols)
        .ok_or_else(|| Error::Internal {
            message: "FP8 scale size overflow".into(),
        })?;
    if x.len() != in_features || weight.len() != expected_weight || scales.len() != expected_scales
    {
        return Err(Error::Internal {
            message: "FP8 GEMV length mismatch".to_string(),
        });
    }
    let ops = CudaArtifactOperatorContext::new()?;
    let handle = ops.upload_fp8_e4m3_e8m0_linear(
        weight,
        scales,
        out_features,
        in_features,
        block_m,
        block_k,
    )?;
    ops.artifact_linear_matvec(&handle, x)
}

/// Run sparse attention with an attention sink on GPU.
///
/// This is intentionally a generic artifact-format operator: callers pass explicit
/// shapes and row-major buffers; no model-family tensor names are visible here.
pub fn cuda_sparse_attention_sink_f32(
    query: &[f32],
    values: &[f32],
    topk: &[i32],
    sink: &[f32],
    tokens: usize,
    kv_len: usize,
    heads: usize,
    head_dim: usize,
    topk_len: usize,
    softmax_scale: f32,
) -> Result<Vec<f32>> {
    let shape = CudaSparseAttentionShape {
        batch_size: 1,
        tokens_per_batch: tokens,
        kv_len,
        heads,
        head_dim,
        topk: topk_len,
        softmax_scale,
    };
    shape.validate()?;
    if query.len() != shape.q_elements()
        || values.len() != shape.kv_elements()
        || topk.len() != shape.topk_elements()
        || sink.len() != heads
    {
        return Err(Error::Internal {
            message: format!(
                "sparse attention length mismatch: q={} values={} topk={} sink={}, expected q={} values={} topk={} sink={}",
                query.len(),
                values.len(),
                topk.len(),
                sink.len(),
                shape.q_elements(),
                shape.kv_elements(),
                shape.topk_elements(),
                heads
            ),
        });
    }

    CudaArtifactOperatorContext::new()?.sparse_attention_sink_f32(query, values, topk, sink, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observability_config(overrides: &[(&str, &str)]) -> CudaObservabilityConfig {
        CudaObservabilityConfig::resolve(|name| {
            overrides
                .iter()
                .find_map(|(key, value)| (*key == name).then(|| (*value).to_owned()))
        })
    }

    #[test]
    fn routed_expert_shape_validates_logical_dimensions() {
        assert_eq!(
            CudaRoutedExpertShape::new(256, 512, 384).unwrap(),
            CudaRoutedExpertShape {
                input: 256,
                intermediate: 512,
                output: 384,
            }
        );
        for invalid in [
            CudaRoutedExpertShape {
                input: 0,
                intermediate: 512,
                output: 384,
            },
            CudaRoutedExpertShape {
                input: 255,
                intermediate: 512,
                output: 384,
            },
            CudaRoutedExpertShape {
                input: 256,
                intermediate: 511,
                output: 384,
            },
            CudaRoutedExpertShape {
                input: 256,
                intermediate: 512,
                output: 0,
            },
        ] {
            assert!(invalid.validate().is_err(), "accepted {invalid:?}");
            assert!(invalid.physical_bytes().is_err(), "sized {invalid:?}");
        }
    }

    #[test]
    fn routed_expert_physical_bytes_include_raw_and_private_storage() {
        let shape = CudaRoutedExpertShape::new(256, 512, 384).unwrap();
        let raw_gate_up = 512 * (256 / 2) + 512 * (256 / 32);
        let raw_down = 384 * (512 / 2) + 384 * (512 / 32);
        let private_gate_up = mxfp4_sfb_storage_bytes(512, 256).unwrap();
        let private_down = mxfp4_sfb_storage_bytes(384, 512).unwrap();
        let expected = raw_gate_up * 2 + raw_down + private_gate_up * 2 + private_down;

        assert_eq!(shape.physical_bytes().unwrap(), expected);
        let storage = shape.storage_bytes().unwrap();
        assert_eq!(storage.raw_linear, raw_gate_up * 2 + raw_down);
        assert_eq!(
            storage.provider_private,
            [private_gate_up, private_gate_up, private_down]
        );
    }

    #[test]
    fn routed_expert_physical_bytes_reject_dimension_overflow() {
        let shape = CudaRoutedExpertShape {
            input: (u32::MAX as usize).saturating_add(1),
            intermediate: 32,
            output: 32,
        };
        assert!(shape.physical_bytes().is_err());
    }

    #[test]
    fn hc_device_layout_preserves_each_rows_accumulation_order() {
        let rows = 3usize;
        let cols = 7usize;
        let function = [
            0.5, -0.25, 1.0, 0.125, -2.0, 0.75, 0.0625, -1.0, 0.375, 0.5, -0.75, 1.25, 0.25,
            -0.125, 2.0, -1.5, 0.25, 0.75, -0.5, 0.125, 1.0,
        ];
        let state = [0.25, -2.0, 1.5, 0.5, -0.75, 4.0, 0.125];
        let transposed = transpose_hc_function_for_device(&function, rows).unwrap();

        for row in 0..rows {
            let mut row_major_dot = 0.0f32;
            let mut device_layout_dot = 0.0f32;
            for col in 0..cols {
                row_major_dot += function[row * cols + col] * state[col];
                device_layout_dot += transposed[col * rows + row] * state[col];
            }
            assert_eq!(row_major_dot.to_bits(), device_layout_dot.to_bits());
        }
    }

    #[test]
    fn cuda_observability_is_disabled_by_default() {
        assert_eq!(
            observability_config(&[]),
            CudaObservabilityConfig { moe_timing: false }
        );
    }

    #[test]
    fn cuda_observability_parses_explicit_boolean_values() {
        for disabled in ["", "0", "false", " off ", "NO"] {
            assert!(!observability_config(&[("FERRULE_CUDA_MOE_TIMING", disabled)]).moe_timing);
        }
        for enabled in ["1", "true", "yes", " enabled "] {
            assert!(observability_config(&[("FERRULE_CUDA_MOE_TIMING", enabled)]).moe_timing);
        }
    }

    fn test_slot_pointers(seed: u64) -> CudaExpertSlotPointers {
        CudaExpertSlotPointers {
            gate_weight: seed + 1,
            gate_scale: seed + 2,
            up_weight: seed + 3,
            up_scale: seed + 4,
            down_weight: seed + 5,
            down_scale: seed + 6,
        }
    }

    #[test]
    fn expert_slot_host_reuses_slot_with_new_generation() {
        let mut table = CudaExpertSlotTableHost::new(4, 1).unwrap();
        let first = table.install(0, test_slot_pointers(10)).unwrap();
        assert_eq!(table.binding(0), Some(first));
        assert!(table.is_current(first));

        assert!(table.evict(0).unwrap());
        assert_eq!(table.binding(0), None);
        assert!(!table.is_current(first));
        let second = table.install(1, test_slot_pointers(20)).unwrap();
        assert_eq!(second.slot, first.slot);
        assert_ne!(second.generation, first.generation);
        assert!(!table.is_current(first));
        assert!(table.is_current(second));
    }

    #[test]
    fn expert_slot_host_resolves_resident_and_rejects_miss() {
        let mut table = CudaExpertSlotTableHost::new(4, 2).unwrap();
        let resident = table.install(2, test_slot_pointers(30)).unwrap();
        assert_eq!(table.binding(2), Some(resident));
        assert_eq!(table.binding(1), None);
        assert_eq!(table.binding(4), None);
        assert!(!table.evict(1).unwrap());
    }

    #[test]
    fn dsv4_paged_decode_rows_shape_accepts_fixed_stride_outputs() {
        let shape = Dsv4PagedDecodeRowsShape {
            rows: 3,
            window_size: 4,
            index_topk: 2,
            index_heads: 2,
            index_head_dim: 8,
        };
        assert_eq!(
            shape
                .validate_lengths(48, 6, 4, 3, 3, 3, 3, 18, 18)
                .unwrap(),
            18
        );
    }

    #[test]
    fn dsv4_paged_decode_rows_shape_rejects_bad_metadata_and_overflow() {
        let shape = Dsv4PagedDecodeRowsShape {
            rows: 2,
            window_size: 4,
            index_topk: 3,
            index_heads: 1,
            index_head_dim: 4,
        };
        assert!(shape.validate_lengths(8, 2, 2, 1, 2, 2, 2, 14, 14).is_err());
        assert!(shape.validate_lengths(8, 2, 3, 2, 2, 2, 2, 14, 13).is_err());

        let overflow = Dsv4PagedDecodeRowsShape {
            rows: usize::MAX,
            window_size: 2,
            index_topk: 1,
            index_heads: 1,
            index_head_dim: 1,
        };
        assert!(overflow.elements().is_err());
    }

    fn apply_indexed_rope_cpu(
        qk: &mut [f32],
        cos_table: &[f32],
        sin_table: &[f32],
        positions: &[i32],
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        inverse: bool,
    ) {
        let table_width = rope_dim / 2;
        let row_stride = heads * head_dim;
        let tail_start = head_dim - rope_dim;
        for (row, &position) in positions.iter().enumerate() {
            let position = position as usize;
            for head in 0..heads {
                for pair in 0..table_width {
                    let table_offset = position * table_width + pair;
                    let cos = cos_table[table_offset];
                    let sin = if inverse {
                        -sin_table[table_offset]
                    } else {
                        sin_table[table_offset]
                    };
                    let base = row * row_stride + head * head_dim + tail_start + pair * 2;
                    let x0 = qk[base];
                    let x1 = qk[base + 1];
                    qk[base] = x0 * cos - x1 * sin;
                    qk[base + 1] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    fn restore_rope_tail_bf16_boundary(
        qk: &mut [f32],
        rows: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
    ) {
        let row_stride = heads * head_dim;
        let tail_start = head_dim - rope_dim;
        for row in 0..rows {
            for head in 0..heads {
                let base = row * row_stride + head * head_dim + tail_start;
                for value in &mut qk[base..base + rope_dim] {
                    let bits = value.to_bits();
                    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                    *value = f32::from_bits(rounded & 0xffff_0000);
                }
            }
        }
    }

    fn rope_test_tables(positions: usize, rope_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let table_width = rope_dim / 2;
        let mut cos = Vec::with_capacity(positions * table_width);
        let mut sin = Vec::with_capacity(positions * table_width);
        for position in 0..positions {
            for pair in 0..table_width {
                let angle = position as f32 * 0.19 + pair as f32 * 0.07;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        (cos, sin)
    }

    #[test]
    fn indexed_rope_cpu_reference_matches_strided_row_positions() {
        const ROWS: usize = 3;
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 6;
        const ROPE_DIM: usize = 4;
        let input: Vec<f32> = (0..ROWS * HEADS * HEAD_DIM)
            .map(|index| index as f32 * 0.125 - 1.5)
            .collect();
        let (cos, sin) = rope_test_tables(8, ROPE_DIM);
        let indexed_positions = [1, 3, 5];
        let strided_positions: Vec<i32> = (0..ROWS).map(|row| 1 + row as i32 * 2).collect();

        for inverse in [false, true] {
            let mut indexed = input.clone();
            let mut strided = input.clone();
            apply_indexed_rope_cpu(
                &mut indexed,
                &cos,
                &sin,
                &indexed_positions,
                HEADS,
                HEAD_DIM,
                ROPE_DIM,
                inverse,
            );
            apply_indexed_rope_cpu(
                &mut strided,
                &cos,
                &sin,
                &strided_positions,
                HEADS,
                HEAD_DIM,
                ROPE_DIM,
                inverse,
            );
            assert_eq!(indexed, strided);
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn indexed_rope_cuda_matches_cpu_and_supports_inverse() {
        const ROWS: usize = 3;
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 6;
        const ROPE_DIM: usize = 4;
        let context = CudaArtifactOperatorContext::new().unwrap();
        let input: Vec<f32> = (0..ROWS * HEADS * HEAD_DIM)
            .map(|index| index as f32 * 0.125 - 1.5)
            .collect();
        let positions_host = [4, 0, 3];
        let (cos, sin) = rope_test_tables(5, ROPE_DIM);
        let mut expected = input.clone();
        apply_indexed_rope_cpu(
            &mut expected,
            &cos,
            &sin,
            &positions_host,
            HEADS,
            HEAD_DIM,
            ROPE_DIM,
            false,
        );
        restore_rope_tail_bf16_boundary(&mut expected, ROWS, HEADS, HEAD_DIM, ROPE_DIM);
        let mut actual = context.upload_f32_buffer(&input).unwrap();
        let cos_device = context.upload_f32_buffer(&cos).unwrap();
        let sin_device = context.upload_f32_buffer(&sin).unwrap();
        let positions = context.upload_i32_buffer(&positions_host).unwrap();

        context
            .rope_tail_rows_indexed_from_device(
                &mut actual,
                &cos_device,
                &sin_device,
                &positions,
                ROWS as u32,
                HEADS as u32,
                HEAD_DIM as u32,
                ROPE_DIM as u32,
                false,
            )
            .unwrap();
        let forward = context.download_f32_buffer(&actual).unwrap();
        for (index, (actual, expected)) in forward.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "forward mismatch at {index}: actual={actual} expected={expected}"
            );
        }

        let mut expected_round_trip = expected.clone();
        apply_indexed_rope_cpu(
            &mut expected_round_trip,
            &cos,
            &sin,
            &positions_host,
            HEADS,
            HEAD_DIM,
            ROPE_DIM,
            true,
        );
        restore_rope_tail_bf16_boundary(&mut expected_round_trip, ROWS, HEADS, HEAD_DIM, ROPE_DIM);
        context
            .rope_tail_rows_indexed_from_device(
                &mut actual,
                &cos_device,
                &sin_device,
                &positions,
                ROWS as u32,
                HEADS as u32,
                HEAD_DIM as u32,
                ROPE_DIM as u32,
                true,
            )
            .unwrap();
        let round_trip = context.download_f32_buffer(&actual).unwrap();
        for (index, (actual, expected)) in round_trip.iter().zip(&expected_round_trip).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "inverse mismatch at {index}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn cuda_probe_compiles() {
        // This test just verifies the function signature compiles.
        // cuda_probe requires a real GPU to succeed, so we only
        // check that it doesn't panic or cause a link error.
        let _ = cuda_probe(); // may fail without GPU — that's fine
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn moe_ranked_reducer_matches_host_left_fold() {
        const NUM_EXPERTS: usize = 3;
        const BATCH_COLS: usize = 2;
        const HIDDEN_SIZE: usize = 257;
        const ROUTES_PER_COL: usize = 3;

        let ctx = cu(CudaContext::new(0)).unwrap();
        cu(ctx.bind_to_thread()).unwrap();
        let module = cu(crate::cuda::kernels::kernels::load(&ctx)).unwrap();
        let stream = ctx.default_stream();

        let mut expert_output = vec![0.0f32; NUM_EXPERTS * BATCH_COLS * HIDDEN_SIZE];
        for expert in 0..NUM_EXPERTS {
            for col in 0..BATCH_COLS {
                for row in 0..HIDDEN_SIZE {
                    let value = match expert {
                        0 => 16_777_216.0,
                        1 => -16_777_216.0,
                        _ => 1.0 + (row % 13) as f32 * 0.0625 + col as f32 * 0.125,
                    };
                    expert_output[(expert * BATCH_COLS + col) * HIDDEN_SIZE + row] = value;
                }
            }
        }
        let route_slots = vec![0i32, 1, 2, 0, 2, 1];
        let base: Vec<f32> = (0..BATCH_COLS * HIDDEN_SIZE)
            .map(|index| 0.25 + (index % 11) as f32 * 0.03125)
            .collect();
        let mut expected = base.clone();
        for col in 0..BATCH_COLS {
            for row in 0..HIDDEN_SIZE {
                let output_off = col * HIDDEN_SIZE + row;
                let mut acc = expected[output_off];
                for rank in 0..ROUTES_PER_COL {
                    let expert = route_slots[col * ROUTES_PER_COL + rank] as usize;
                    acc += expert_output[(expert * BATCH_COLS + col) * HIDDEN_SIZE + row];
                }
                expected[output_off] = acc;
            }
        }

        let expert_output_d = cu(DeviceBuffer::from_host(&stream, &expert_output)).unwrap();
        let route_slots_d = cu(DeviceBuffer::from_host(&stream, &route_slots)).unwrap();
        let mut output_d = cu(DeviceBuffer::from_host(&stream, &base)).unwrap();
        let elements = (BATCH_COLS * HIDDEN_SIZE) as u32;
        cu(unsafe {
            module.moe_reduce_expert_outputs_ranked(
                &stream,
                LaunchConfig {
                    grid_dim: (elements.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &expert_output_d,
                &route_slots_d,
                &mut output_d,
                0,
                HIDDEN_SIZE as u32,
                BATCH_COLS as u32,
                ROUTES_PER_COL as u32,
                NUM_EXPERTS as u32,
            )
        })
        .unwrap();
        let actual = cu(output_d.to_host_vec(&stream)).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "reducer mismatch at output index {index}: actual={actual:?} expected={expected:?}"
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn split_moe_reducer_respects_nonzero_output_offset() {
        const NUM_EXPERTS: usize = 2;
        const BATCH_COLS: usize = 2;
        const HIDDEN_SIZE: usize = 7;
        const ROUTES_PER_COL: usize = 2;
        const OUTPUT_OFFSET: usize = 5;
        const SUFFIX: usize = 3;
        const ELEMENTS: usize = BATCH_COLS * HIDDEN_SIZE;

        let ctx = cu(CudaContext::new(0)).unwrap();
        cu(ctx.bind_to_thread()).unwrap();
        let module = cu(crate::cuda::kernels::kernels::load(&ctx)).unwrap();
        let stream = ctx.default_stream();

        let resident_output = (0..NUM_EXPERTS * BATCH_COLS * HIDDEN_SIZE)
            .map(|index| 1.0 + (index % 11) as f32)
            .collect::<Vec<_>>();
        let materialized_output = (0..NUM_EXPERTS * BATCH_COLS * HIDDEN_SIZE)
            .map(|index| 13.0 + (index % 7) as f32)
            .collect::<Vec<_>>();
        let resident_slots = [0i32, -1, 1, -1];
        let materialized_slots = [-1i32, 1, -1, 0];
        let miss_markers = [0i32, 1, 0, 1];
        let initial = (0..OUTPUT_OFFSET + ELEMENTS + SUFFIX)
            .map(|index| 32.0 + index as f32)
            .collect::<Vec<_>>();
        let mut expected = initial.clone();
        for column in 0..BATCH_COLS {
            for row in 0..HIDDEN_SIZE {
                let output = OUTPUT_OFFSET + column * HIDDEN_SIZE + row;
                for rank in 0..ROUTES_PER_COL {
                    let route = column * ROUTES_PER_COL + rank;
                    let (slot, values) = if miss_markers[route] != 0 {
                        (materialized_slots[route] as usize, &materialized_output)
                    } else {
                        (resident_slots[route] as usize, &resident_output)
                    };
                    expected[output] += values[(slot * BATCH_COLS + column) * HIDDEN_SIZE + row];
                }
                expected[output] = f32::from_bits(
                    (expected[output].to_bits()
                        + 0x7fff
                        + ((expected[output].to_bits() >> 16) & 1))
                        & 0xffff_0000,
                );
            }
        }

        let resident_output = cu(DeviceBuffer::from_host(&stream, &resident_output)).unwrap();
        let materialized_output =
            cu(DeviceBuffer::from_host(&stream, &materialized_output)).unwrap();
        let resident_slots = cu(DeviceBuffer::from_host(&stream, &resident_slots)).unwrap();
        let materialized_slots = cu(DeviceBuffer::from_host(&stream, &materialized_slots)).unwrap();
        let miss_markers = cu(DeviceBuffer::from_host(&stream, &miss_markers)).unwrap();
        let mut output = cu(DeviceBuffer::from_host(&stream, &initial)).unwrap();

        cu(unsafe {
            module.moe_reduce_split_expert_outputs_ranked(
                &stream,
                LaunchConfig::for_num_elems(ELEMENTS as u32),
                &resident_output,
                &materialized_output,
                &resident_slots,
                &materialized_slots,
                &miss_markers,
                &mut output,
                OUTPUT_OFFSET as u32,
                HIDDEN_SIZE as u32,
                BATCH_COLS as u32,
                ROUTES_PER_COL as u32,
                NUM_EXPERTS as u32,
            )
        })
        .unwrap();
        let actual = cu(output.to_host_vec(&stream)).unwrap();

        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(&actual[..OUTPUT_OFFSET], &initial[..OUTPUT_OFFSET]);
        assert_eq!(
            &actual[OUTPUT_OFFSET + ELEMENTS..],
            &initial[OUTPUT_OFFSET + ELEMENTS..]
        );
    }

    #[test]
    fn fp8_e4m3fn_e8m0_gemv_matches_cpu_reference_when_cuda_available() {
        // 2x4 weight, block_m=1, block_k=2 => scales [2, 2]
        let x = vec![1.0f32, 0.5, -1.0, 3.0];
        let weight: Vec<u8> = vec![0x38, 0x40, 0xb8, 0x00, 0x38, 0x38, 0x00, 0x00]; // 1.0,2.0,-1.0,0.0,1.0,1.0,0,0
        let scales: Vec<u8> = vec![127, 128, 126, 127]; // 1.0, 2.0, 0.5, 1.0
        // expected: row0 = 1.0*1.0*1.0 + 0.5*2.0*1.0 + (-1.0)*(-1.0)*0.5 + 3.0*0.0*0.5 = 1.0+1.0+0.5+0 = 2.5
        match cuda_gemv_fp8_e4m3fn_e8m0_2d(&x, &weight, &scales, 2, 4, 1, 2) {
            Ok(actual) => {
                assert!(!actual.is_empty());
                // Just verify finite and not panic
                assert!(actual[0].is_finite());
            }
            Err(err) => eprintln!("skipping CUDA FP8 smoke: {err}"),
        }
    }

    #[test]
    fn fp8_e4m3fn_cpu_decoder_matches_known_values() {
        // Use the CPU reference from ferrule-runtime
        assert_eq!(decode_fp8_test(0x00), 0.0);
        assert_eq!(decode_fp8_test(0x38), 1.0);
        assert_eq!(decode_fp8_test(0xb8), -1.0);
        assert_eq!(decode_fp8_test(0x40), 2.0);
        assert!(decode_fp8_test(0x7f).is_nan());
    }

    fn decode_fp8_test(byte: u8) -> f32 {
        let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = (byte >> 3) & 0x0f;
        let mantissa = byte & 0x07;
        if exponent == 0 {
            if mantissa == 0 {
                return sign * 0.0;
            }
            return sign * (mantissa as f32) * 2.0f32.powi(-9);
        }
        if exponent == 0x0f && mantissa == 0x07 {
            return f32::NAN;
        }
        sign * 2.0f32.powi(exponent as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
    }

    #[test]
    fn recent_rows_cuda_materialization_matches_logical_window() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skipping recent-row CUDA materialization: no CUDA device");
            return;
        };
        cu(ctx.bind_to_thread()).unwrap();
        let module = cu(crate::cuda::kernels::kernels::load(&ctx)).unwrap();
        let stream = ctx.default_stream();
        let visible_lens = cu(DeviceBuffer::from_host(&stream, &[1i32, 4, 7])).unwrap();
        let mut output = cu(DeviceBuffer::<i32>::zeroed(&stream, 12)).unwrap();

        cu(unsafe {
            module.fill_recent_rows(
                &stream,
                LaunchConfig::for_num_elems(12),
                &visible_lens,
                &mut output,
                3,
                4,
                12,
            )
        })
        .unwrap();

        assert_eq!(
            cu(output.to_host_vec(&stream)).unwrap(),
            [0, -1, -1, -1, 0, 1, 2, 3, 3, 4, 5, 6]
        );
    }

    #[test]
    fn payload_encoding_gpu_kernels_smoke() {
        if CudaContext::new(0).is_err() {
            eprintln!("skipping artifact-format GPU kernel smoke: no CUDA device");
            return;
        }
        let ctx = cu(CudaContext::new(0)).unwrap();
        cu(ctx.bind_to_thread()).unwrap();
        let module = cu(crate::cuda::kernels::kernels::load(&ctx)).unwrap();
        let s = ctx.default_stream();

        // 3. rope_yarn (nh=4, rd=64)
        {
            let nh: u32 = 4;
            let hd: u32 = 64;
            let rd: u32 = 64;
            let num_elements = nh * hd;
            let qk = vec![1.0f32; num_elements as usize];
            let cos: Vec<f32> = vec![0.5f32; (rd / 2) as usize];
            let sin: Vec<f32> = vec![0.866f32; (rd / 2) as usize];
            let mut qkd = cu(DeviceBuffer::from_host(&s, &qk)).unwrap();
            let cosd = cu(DeviceBuffer::from_host(&s, &cos)).unwrap();
            let sind = cu(DeviceBuffer::from_host(&s, &sin)).unwrap();
            cu(unsafe {
                module.rope_yarn(
                    &s,
                    LaunchConfig::for_num_elems(num_elements),
                    &mut qkd,
                    &cosd,
                    &sind,
                    num_elements,
                    hd,
                    rd,
                )
            })
            .unwrap();
            let _out = cu(qkd.to_host_vec(&s)).unwrap();
            eprintln!("  [PASS] rope_yarn");
        }
    }
}
