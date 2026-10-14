//! Synchronous streaming/fixed-split GQA, reconstructed from the final delivery
//! contract. No runtime placement/controller or live-cache lease authority.
//! Device partials are merged and normalized on host, not GPU-resident output.

use super::{CudaF32Buffer, CudaOperators, CudaStream, cu, runtime};
use crate::cpu::streaming_attention::{
    streaming_error, validate_kv_nonoverlap, validate_positions,
};
pub use crate::cpu::{
    StreamingF32GqaConfig, StreamingF32GqaOwnedKv, StreamingF32GqaProtection,
    StreamingF32GqaRequest,
};
use crate::cpu::{StreamingF32GqaPartial, streaming_f32_gqa_partial};
use crate::cuda::ffi::core::StreamingGqaArgs;
use ferrule_common::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone)]
pub struct StreamingF32GqaStats {
    pub cpu: Duration,
    pub query_d2h: Duration,
    pub query_h2d: Duration,
    pub kv_h2d: Duration,
    pub gpu_compute_and_partial_d2h: Duration,
    pub merge: Duration,
    pub output_h2d: Duration,
    pub gpu_blocks: usize,
    pub kv_h2d_bytes: usize,
}
#[derive(Debug)]
pub struct StreamingF32GqaResult {
    pub output: Vec<f32>,
    pub stats: StreamingF32GqaStats,
}
pub struct CudaStreamingF32GqaRequest<'a> {
    pub config: StreamingF32GqaConfig,
    pub query: &'a CudaF32Buffer,
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub query_start: u64,
    pub kv_start: u64,
}

/// Two reusable device arenas: Q + partial, and current K + V block.
/// Budget charges allocator size classes, not idle allocator segments, CUDA
/// context/modules, caller Q/output or authoritative host KV. Host scratch is
/// also bounded by this geometry. No pinned asynchronous overlap is claimed.
pub struct CudaStreamingF32GqaWorkspace {
    config: StreamingF32GqaConfig,
    arenas: Option<[CudaF32Buffer; 2]>,
    stream: Arc<CudaStream>,
    scratch: Vec<f32>,
    bytes: usize,
    poisoned: bool,
    unknown: bool,
}
impl CudaStreamingF32GqaWorkspace {
    pub fn required_bytes(config: StreamingF32GqaConfig) -> Result<usize> {
        let [a, b] = arena_lengths(config)?;
        granted(a)?
            .checked_add(granted(b)?)
            .ok_or_else(|| streaming_error("workspace budget overflow"))
    }
    pub fn allocated_bytes(&self) -> usize {
        self.bytes
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn host_scratch_bytes(&self) -> usize {
        self.scratch.len() * 4
    }
}
impl Drop for CudaStreamingF32GqaWorkspace {
    fn drop(&mut self) {
        // Unknown is not completion: retain DMA/kernel operands if the exact
        // owner stream cannot establish quiescence even during cleanup.
        if self.unknown && self.stream.synchronize().is_err() {
            std::mem::forget(self.arenas.take());
            std::mem::forget(std::mem::take(&mut self.scratch));
            std::mem::forget(Arc::clone(&self.stream));
        }
    }
}
fn arena_lengths(c: StreamingF32GqaConfig) -> Result<[usize; 2]> {
    c.validate()?;
    for n in [c.rows, c.q_heads, c.kv_heads, c.head_dim, c.block_tokens] {
        u32::try_from(n).map_err(|_| streaming_error("geometry exceeds CUDA u32 ABI"))?;
    }
    if c.rows * c.q_heads > i32::MAX as usize {
        return Err(streaming_error("CUDA grid limit"));
    }
    let a = c.query_elements()?.checked_add(c.partial_elements()?);
    let b = c.kv_elements(c.block_tokens)?.checked_mul(2);
    match (a, b) {
        (Some(a), Some(b)) if a <= isize::MAX as usize / 4 && b <= isize::MAX as usize / 4 => {
            Ok([a, b])
        }
        _ => Err(streaming_error("workspace size overflow")),
    }
}
// Mirrors the ordinary allocator's small power-of-two and large 2 MiB classes.
// Arenas explicitly request this rounded extent rather than charging payload.
fn granted(elements: usize) -> Result<usize> {
    let bytes = elements
        .checked_mul(4)
        .ok_or_else(|| streaming_error("byte overflow"))?;
    if bytes <= 1 << 20 {
        Ok(bytes.max(256).next_power_of_two())
    } else {
        bytes
            .checked_add((2 << 20) - 1)
            .map(|n| n & !((2 << 20) - 1))
            .ok_or_else(|| streaming_error("allocation class overflow"))
    }
}

impl CudaOperators {
    pub fn streaming_f32_gqa_workspace(
        &self,
        config: StreamingF32GqaConfig,
        budget_bytes: usize,
    ) -> Result<CudaStreamingF32GqaWorkspace> {
        self.check_capture_safe("streaming attention workspace")?;
        let bytes = CudaStreamingF32GqaWorkspace::required_bytes(config)?;
        if bytes > budget_bytes {
            return Err(streaming_error("workspace budget exceeded"));
        }
        let [a, b] = arena_lengths(config)?;
        let arenas = [
            self.zero_f32_buffer(granted(a)? / 4)?,
            self.zero_f32_buffer(granted(b)? / 4)?,
        ];
        let scratch_len = config
            .query_elements()?
            .max(config.partial_elements()?)
            .max(config.kv_elements(config.block_tokens)?);
        Ok(CudaStreamingF32GqaWorkspace {
            config,
            arenas: Some(arenas),
            stream: Arc::clone(&self.stream),
            scratch: vec![0.0; scratch_len],
            bytes,
            poisoned: false,
            unknown: false,
        })
    }
    fn streaming_precheck(
        &self,
        config: StreamingF32GqaConfig,
        query_len: usize,
        key: &[f32],
        value: &[f32],
        query_start: u64,
        kv_start: u64,
        cpu_tokens: usize,
        protection: &dyn StreamingF32GqaProtection,
        w: &CudaStreamingF32GqaWorkspace,
    ) -> Result<usize> {
        self.check_capture_safe("synchronous streaming attention")?;
        if self._ctx.is_capturing() {
            return Err(streaming_error("graph capture is unsupported"));
        }
        if w.poisoned {
            return Err(streaming_error("workspace is poisoned"));
        }
        if !Arc::ptr_eq(&w.stream, &self.stream) {
            return Err(streaming_error("workspace owner mismatch"));
        }
        config.validate()?;
        if config != w.config
            || query_len != config.query_elements()?
            || key.len() != value.len()
            || !key.len().is_multiple_of(config.kv_elements(1)?)
        {
            return Err(streaming_error("request/workspace shape mismatch"));
        }
        let tokens = key.len() / config.kv_elements(1)?;
        if cpu_tokens > tokens {
            return Err(streaming_error("CPU split exceeds history"));
        }
        validate_positions(query_start, config.rows)?;
        validate_positions(kv_start, tokens)?;
        validate_kv_nonoverlap(key, value)?;
        protection.validate(key, value)?;
        Ok(tokens)
    }
    /// CPU prefix plus GPU-streamed suffix. Each block finishes before staging
    /// reuse; protection stays borrowed through all synchronous consumption.
    pub fn streaming_f32_gqa_fixed_split(
        &self,
        request: StreamingF32GqaRequest<'_>,
        cpu_tokens: usize,
        protection: &dyn StreamingF32GqaProtection,
        workspace: &mut CudaStreamingF32GqaWorkspace,
    ) -> Result<StreamingF32GqaResult> {
        self.streaming_precheck(
            request.config,
            request.query.len(),
            request.key,
            request.value,
            request.query_start,
            request.kv_start,
            cpu_tokens,
            protection,
            workspace,
        )?;
        let result = self.streaming_execute(request, cpu_tokens, workspace);
        if result.is_err() {
            workspace.poisoned = true;
        }
        result
    }
    /// Explicit device Q D2H and final host-normalized output H2D boundary.
    pub fn streaming_f32_gqa_fixed_split_into(
        &self,
        request: CudaStreamingF32GqaRequest<'_>,
        cpu_tokens: usize,
        protection: &dyn StreamingF32GqaProtection,
        output: &mut CudaF32Buffer,
        w: &mut CudaStreamingF32GqaWorkspace,
    ) -> Result<StreamingF32GqaStats> {
        self.streaming_precheck(
            request.config,
            request.query.len(),
            request.key,
            request.value,
            request.query_start,
            request.kv_start,
            cpu_tokens,
            protection,
            w,
        )?;
        self.check_buffer_owner(&request.query.buffer, "streaming query")?;
        self.check_buffer_owner(&output.buffer, "streaming output")?;
        let arenas = w.arenas.as_ref().expect("live workspace");
        if output.len() != request.config.query_elements()? {
            return Err(streaming_error("device extent mismatch"));
        }
        Self::f32_no_alias(&[request.query], &[output])?;
        Self::f32_no_alias(&[request.query, output], &[&arenas[0], &arenas[1]])?;
        let result = (|| {
            let start = Instant::now();
            self.streaming_download(request.query.buffer.cu_deviceptr(), request.query.len(), w)?;
            let query = w.scratch[..request.query.len()].to_vec();
            let query_d2h = start.elapsed();
            let mut result = self.streaming_execute(
                StreamingF32GqaRequest {
                    config: request.config,
                    query: &query,
                    key: request.key,
                    value: request.value,
                    query_start: request.query_start,
                    kv_start: request.kv_start,
                },
                cpu_tokens,
                w,
            )?;
            result.stats.query_d2h = query_d2h;
            let start = Instant::now();
            self.streaming_upload(&result.output, output.buffer.cu_deviceptr(), w)?;
            result.stats.output_h2d = start.elapsed();
            Ok(result.stats)
        })();
        if result.is_err() {
            w.poisoned = true;
            if w.unknown {
                // Retain external operands too: returning Err is not a DMA ACK.
                if let Ok(buffer) = request.query.buffer.slice(0, request.query.len()) {
                    std::mem::forget(buffer);
                }
                if let Ok(buffer) = output.buffer.slice(0, output.len()) {
                    std::mem::forget(buffer);
                }
            }
        }
        result
    }
    fn streaming_execute(
        &self,
        request: StreamingF32GqaRequest<'_>,
        cpu_tokens: usize,
        w: &mut CudaStreamingF32GqaWorkspace,
    ) -> Result<StreamingF32GqaResult> {
        let c = request.config;
        let tokens = request.validate()?;
        let mut stats = StreamingF32GqaStats::default();
        let start = Instant::now();
        let mut total = streaming_f32_gqa_partial(request.token_range(0, cpu_tokens)?)?;
        stats.cpu = start.elapsed();
        let arenas = w.arenas.as_ref().expect("live workspace");
        let qptr = arenas[0].buffer.cu_deviceptr();
        let pptr = qptr + c.query_elements()? as u64 * 4;
        let kptr = arenas[1].buffer.cu_deviceptr();
        let vptr = kptr + c.kv_elements(c.block_tokens)? as u64 * 4;
        if cpu_tokens < tokens {
            let start = Instant::now();
            self.streaming_upload(request.query, qptr, w)?;
            stats.query_h2d = start.elapsed();
        }
        for begin in (cpu_tokens..tokens).step_by(c.block_tokens) {
            let end = begin.saturating_add(c.block_tokens).min(tokens);
            let block = request.token_range(begin, end)?;
            let start = Instant::now();
            self.streaming_upload(block.key, kptr, w)?;
            self.streaming_upload(block.value, vptr, w)?;
            stats.kv_h2d += start.elapsed();
            stats.kv_h2d_bytes += (block.key.len() + block.value.len()) * 4;
            let start = Instant::now();
            let launched = self.launch_streaming_partial(
                c,
                request.query_start,
                block.kv_start,
                end - begin,
                qptr,
                kptr,
                vptr,
                pptr,
            );
            self.streaming_completion(launched, w)?;
            self.record_kernel_launches(1);
            self.streaming_download(pptr, c.partial_elements()?, w)?;
            stats.gpu_compute_and_partial_d2h += start.elapsed();
            let start = Instant::now();
            let partial = StreamingF32GqaPartial::from_state(
                c,
                request.query_start,
                w.scratch[..c.partial_elements()?].to_vec(),
            )?;
            total.merge(&partial)?;
            stats.merge += start.elapsed();
            stats.gpu_blocks += 1;
        }
        let start = Instant::now();
        let output = total.output();
        stats.merge += start.elapsed();
        Ok(StreamingF32GqaResult { output, stats })
    }
    /// Private typed seam for the committed placement owner. It keeps every
    /// buffer until its recorded consumer event completes (including errors).
    pub(crate) fn streaming_staged_partial(
        &self,
        c: StreamingF32GqaConfig,
        query: &CudaF32Buffer,
        key: &CudaF32Buffer,
        value: &CudaF32Buffer,
        partial: &mut CudaF32Buffer,
        tokens: usize,
        query_start: u64,
        kv_start: u64,
    ) -> Result<()> {
        self.check_capture_safe("committed staged GQA")?;
        if self._ctx.is_capturing() {
            return Err(streaming_error(
                "committed GQA graph capture is unsupported",
            ));
        }
        c.validate()?;
        arena_lengths(c)?;
        validate_positions(query_start, c.rows)?;
        validate_positions(kv_start, tokens)?;
        if tokens == 0
            || tokens > c.block_tokens
            || query.len() != c.query_elements()?
            || key.len() < c.kv_elements(tokens)?
            || value.len() < c.kv_elements(tokens)?
            || partial.len() != c.partial_elements()?
        {
            return Err(streaming_error("staged partial extent mismatch"));
        }
        for buffer in [query, key, value, &*partial] {
            self.check_buffer_owner(&buffer.buffer, "committed partial")?;
        }
        Self::f32_no_alias(&[query, key, value], &[partial])?;
        let result = self.launch_streaming_partial(
            c,
            query_start,
            kv_start,
            tokens,
            query.buffer.cu_deviceptr(),
            key.buffer.cu_deviceptr(),
            value.buffer.cu_deviceptr(),
            partial.buffer.cu_deviceptr(),
        );
        self.record_kernel_launches(1);
        result
    }
    // Both callers validate geometry/owners/aliasing and retain their operands
    // through a real stream/event fence. This is the existing kernel ABI seam.
    fn launch_streaming_partial(
        &self,
        c: StreamingF32GqaConfig,
        query_start: u64,
        kv_start: u64,
        tokens: usize,
        qptr: u64,
        kptr: u64,
        vptr: u64,
        pptr: u64,
    ) -> Result<()> {
        // SAFETY: private callers establish the checked extents and custody above.
        cu(unsafe {
            self.module.streaming_gqa(
                &self.stream,
                StreamingGqaArgs {
                    rows: c.rows as u32,
                    q_heads: c.q_heads as u32,
                    kv_heads: c.kv_heads as u32,
                    head_dim: c.head_dim as u32,
                    tokens: tokens as u32,
                    softmax_scale: c.softmax_scale,
                    query: qptr,
                    key: kptr,
                    value: vptr,
                    partial: pptr,
                    query_start,
                    kv_start,
                    window: c.sliding_window.unwrap_or(0),
                    stream: 0,
                },
            )
        })
    }
    fn streaming_completion(
        &self,
        submitted: Result<()>,
        w: &mut CudaStreamingF32GqaWorkspace,
    ) -> Result<()> {
        // Submission failure can leave work queued: always attempt completion.
        let completion = self.record_stream_wide_sync(|| self.stream.synchronize());
        if completion.is_err() {
            w.unknown = true;
            w.poisoned = true;
        }
        submitted.and(completion)
    }
    fn streaming_upload(
        &self,
        values: &[f32],
        dst: u64,
        w: &mut CudaStreamingF32GqaWorkspace,
    ) -> Result<()> {
        if values.len() > w.scratch.len() {
            return Err(streaming_error("host scratch overflow"));
        }
        // Never submit borrowed host memory: unknown DMA retains owned scratch.
        w.scratch[..values.len()].copy_from_slice(values);
        let submitted = cu(runtime::copy_host_to_device(
            &self.stream,
            dst,
            w.scratch.as_ptr().cast(),
            values.len() * 4,
        ));
        self.streaming_completion(submitted, w)?;
        self.counters.add_host_to_device(values.len() as u64 * 4);
        Ok(())
    }
    fn streaming_download(
        &self,
        src: u64,
        len: usize,
        w: &mut CudaStreamingF32GqaWorkspace,
    ) -> Result<()> {
        if len > w.scratch.len() {
            return Err(streaming_error("host scratch overflow"));
        }
        let submitted = cu(runtime::copy_device_to_host(
            &self.stream,
            w.scratch.as_mut_ptr().cast(),
            src,
            len * 4,
        ));
        self.streaming_completion(submitted, w)?;
        self.counters.add_device_to_host(len as u64 * 4);
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/cuda/streaming_attention.rs"]
mod tests;
