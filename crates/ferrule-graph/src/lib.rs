//! Ferrule Graph — lazy compute graph engine (Rust-native ggml).
//!
//! ## Architecture
//!
//! Like ggml, we use a lazy computation graph:
#![allow(unsafe_code)]
//! 1. **Build phase**: Create tensors, wire operations → a `CGraph`
//! 2. **Compute phase**: Call `graph.compute()` → executes all ops
//!
//! Unlike ggml, our graph is **persistent** — tensors and ops stay on GPU,
//! and subsequent tokens only need to update the input embedding tensor.
//!
//! ## Design
//!
//! ```text
//! Tensor { shape, dtype, data: Storage }
//!   ↓
//! Op { kind, inputs[0..2], output }
//!   ↓
//! CGraph { ops: Vec<Op>, tensors: Arena }
//!   ↓
//! Backend (CPU/SIMD/CUDA) — schedules and executes
//! ```

use ferrule_core::{Error, Result};
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod storage;

// ---------------------------------------------------------------------------
// Tensor shape and storage
// ---------------------------------------------------------------------------

/// Shape descriptor — up to 4D (ggml convention: [ne0, ne1, ne2, ne3]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub dims: [usize; 4],
}

impl Shape {
    pub const fn new(d0: usize, d1: usize, d2: usize, d3: usize) -> Self {
        Self {
            dims: [d0, d1, d2, d3],
        }
    }

    pub const fn scalar() -> Self {
        Self::new(1, 1, 1, 1)
    }

    pub const fn vector(n: usize) -> Self {
        Self::new(n, 1, 1, 1)
    }

    pub const fn matrix(rows: usize, cols: usize) -> Self {
        Self::new(cols, rows, 1, 1)
    }

    pub fn nelements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.dims
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &d)| d > 1)
            .map(|(i, _)| i + 1)
            .unwrap_or(1)
    }
}

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::I32 => 4,
        }
    }
}

/// Owned tensor storage.
#[derive(Debug, Clone)]
pub enum Storage {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    I32(Vec<i32>),
    /// Quantized block storage (from GGUF).
    /// Format: Q4_0, Q4_K, IQ2_XXS, etc.
    Quantized {
        qtype: ferrule_core::QuantType,
        blocks: Vec<u8>,
    },
    /// External buffer (borrowed, not owned by the graph).
    Borrowed {
        ptr: *const u8,
        len: usize,
        dtype: DType,
    },
}

// Safety: Storage is Send + Sync because we ensure no concurrent mutable access.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

/// A tensor node in the compute graph.
#[derive(Debug)]
pub struct Tensor {
    pub shape: Shape,
    pub dtype: DType,
    pub storage: Storage,
    /// Unique id within the arena.
    pub id: usize,
}

/// Operation kinds — mirrors ggml's op enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// y = x * W^T (matrix multiply, GEMV for batch=1)
    MatMul,
    /// RMS normalization: y = x / RMS(x) * weight
    RmsNorm,
    /// Rotary position embedding
    RoPE,
    /// SwiGLU: gate = SiLU(x @ W_gate), up = x @ W_up, y = gate * up @ W_down
    SwiGlu,
    /// SiLU activation
    SiLU,
    /// Softmax along last dim
    Softmax,
    /// Element-wise add (residual)
    Add,
    /// Element-wise multiply
    Mul,
    /// Reshape (view)
    Reshape,
    /// Transpose last two dims
    Transpose,
    /// Slice along a dimension
    Slice,
    /// Copy/conversion between dtypes
    Copy,
    /// Embedding lookup: y = W[ids]
    Embedding,
    /// Argmax along last dim
    Argmax,
}

/// One operation in the compute graph.
#[derive(Debug)]
pub struct Op {
    pub kind: OpKind,
    /// Input tensor indices (0, 1, or 2 operands).
    pub inputs: Vec<usize>,
    /// Output tensor index.
    pub output: usize,
    /// Optional scalar parameter (e.g. epsilon for RMS norm).
    pub param: Option<f32>,
    /// Optional integer parameter (e.g. dim for slice).
    pub iparam: Option<usize>,
}

/// A lazily-built compute graph.
pub struct CGraph {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    /// Which tensors are "inputs" (updated between compute calls).
    pub inputs: Vec<usize>,
    /// Which tensor is the final output.
    pub output: Option<usize>,
}

/// CPU backend for executing compute graphs.
pub struct CpuBackend {
    pub n_threads: usize,
    scratch: Vec<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// CGraph construction
// ---------------------------------------------------------------------------

impl CGraph {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            ops: Vec::new(),
            inputs: Vec::new(),
            output: None,
        }
    }

    /// Add a tensor with owned f32 storage.
    pub fn add_tensor_f32(&mut self, shape: Shape, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            shape,
            dtype: DType::F32,
            storage: Storage::F32(data),
            id,
        });
        id
    }

    /// Add a quantized tensor (from GGUF).
    pub fn add_tensor_quantized(
        &mut self,
        shape: Shape,
        qtype: ferrule_core::QuantType,
        blocks: Vec<u8>,
    ) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            shape,
            dtype: DType::F32, // output dtype after dequant
            storage: Storage::Quantized { qtype, blocks },
            id,
        });
        id
    }

    /// Mark a tensor as an input (will be updated on each compute).
    pub fn mark_input(&mut self, id: usize) {
        self.inputs.push(id);
    }

    /// Set the output tensor.
    pub fn set_output(&mut self, id: usize) {
        self.output = Some(id);
    }

    /// Add an operation. Returns the output tensor id.
    pub fn add_op(
        &mut self,
        kind: OpKind,
        inputs: Vec<usize>,
        shape: Shape,
        param: Option<f32>,
        iparam: Option<usize>,
    ) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            shape,
            dtype: DType::F32,
            storage: Storage::F32(vec![0.0; shape.nelements()]),
            id,
        });
        self.ops.push(Op {
            kind,
            inputs,
            output: id,
            param,
            iparam,
        });
        id
    }

    /// Total memory in bytes used by all tensors.
    pub fn memory_bytes(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| match &t.storage {
                Storage::F32(v) => v.len() * 4,
                Storage::F16(v) => v.len() * 2,
                Storage::Bf16(v) => v.len() * 2,
                Storage::I32(v) => v.len() * 4,
                Storage::Quantized { blocks, .. } => blocks.len(),
                Storage::Borrowed { len, .. } => *len,
            })
            .sum()
    }
}

impl Default for CGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CPU Backend
// ---------------------------------------------------------------------------

impl CpuBackend {
    pub fn new(n_threads: usize) -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .ok();
        Self {
            n_threads,
            scratch: Vec::new(),
        }
    }

    /// Execute all ops in the graph.
    pub fn compute(&mut self, graph: &CGraph) -> Result<()> {
        self.scratch.resize(graph.tensors.len(), Vec::new());

        for op in &graph.ops {
            match op.kind {
                OpKind::MatMul => self.exec_matmul(graph, op)?,
                OpKind::RmsNorm => self.exec_rms_norm(graph, op)?,
                OpKind::SiLU => self.exec_silu(graph, op)?,
                OpKind::Mul => self.exec_mul(graph, op)?,
                OpKind::Add => self.exec_add(graph, op)?,
                OpKind::Softmax => self.exec_softmax(graph, op)?,
                OpKind::Embedding => self.exec_embedding(graph, op)?,
                OpKind::Argmax => self.exec_argmax(graph, op)?,
                OpKind::Reshape => self.exec_reshape(graph, op)?,
                OpKind::Transpose => self.exec_transpose(graph, op)?,
                OpKind::SwiGlu => self.exec_swiglu(graph, op)?,
                _ => {
                    // Unimplemented ops are no-ops for now
                    tracing::warn!("unimplemented op: {:?}", op.kind);
                }
            }
        }

        Ok(())
    }

    /// Get the f32 data for an output tensor.
    pub fn tensor_data_f32<'a>(&'a self, graph: &'a CGraph, id: usize) -> Result<&'a [f32]> {
        let t = &graph.tensors[id];
        match &t.storage {
            Storage::F32(v) => Ok(v),
            _ => Err(Error::Graph("tensor is not f32".into())),
        }
    }

    /// Resolve tensor data to f32, dequantizing if needed.
    fn resolve_f32<'a>(&'a self, graph: &'a CGraph, id: usize) -> Result<&'a [f32]> {
        let t = &graph.tensors[id];
        match &t.storage {
            Storage::F32(v) => Ok(v),
            _ => {
                if !self.scratch[id].is_empty() {
                    return Ok(&self.scratch[id]);
                }
                Err(Error::Graph(format!(
                    "tensor {id} is not f32 and not cached"
                )))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Op implementations
    // -----------------------------------------------------------------------

    fn exec_matmul(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        // y = x @ W^T
        // x: [M, K], W: [N, K] (stored), W^T: [K, N] → result: [M, N]
        let x = self.resolve_f32(graph, op.inputs[0])?;
        let w = self.resolve_f32(graph, op.inputs[1])?;

        let x_shape = &graph.tensors[op.inputs[0]].shape;
        let w_shape = &graph.tensors[op.inputs[1]].shape;

        // Treat x as [M, K], w as [N, K]
        let m = x_shape.dims[1].max(1); // batch
        let k = x_shape.dims[0]; // inner
        let n = w_shape.dims[0]; // out features

        let mut out = vec![0.0f32; m * n];

        // Parallelize over rows if batch is large enough
        if m > 4 {
            use rayon::prelude::*;
            out.par_chunks_mut(n)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let x_row = &x[row * k..(row + 1) * k];
                    for (j, out_j) in out_row.iter_mut().enumerate() {
                        let mut dot = 0.0f32;
                        let w_row = &w[j * k..(j + 1) * k];
                        for p in 0..k {
                            dot += x_row[p] * w_row[p];
                        }
                        *out_j = dot;
                    }
                });
        } else {
            for row in 0..m {
                let x_row = &x[row * k..(row + 1) * k];
                for j in 0..n {
                    let mut dot = 0.0f32;
                    let w_row = &w[j * k..(j + 1) * k];
                    for p in 0..k {
                        dot += x_row[p] * w_row[p];
                    }
                    out[row * n + j] = dot;
                }
            }
        }

        let tid = op.output;
        self.scratch[tid] = out;
        Ok(())
    }

    fn exec_rms_norm(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        let x = self.resolve_f32(graph, op.inputs[0])?;
        let w = self.resolve_f32(graph, op.inputs[1])?;
        let eps = op.param.unwrap_or(1e-5);

        let n = x.len();
        let norm_dim = w.len();
        let n_groups = n / norm_dim;

        let mut out = vec![0.0f32; n];
        for g in 0..n_groups {
            let start = g * norm_dim;
            let end = start + norm_dim;
            let ms = x[start..end].iter().map(|&v| v * v).sum::<f32>() / norm_dim as f32;
            let rms = 1.0 / (ms + eps).sqrt();
            for (j, &xj) in x[start..end].iter().enumerate() {
                out[start + j] = xj * rms * w[j];
            }
        }

        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_silu(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        let x = self.resolve_f32(graph, op.inputs[0])?;
        let mut out = vec![0.0f32; x.len()];
        for (i, &v) in x.iter().enumerate() {
            out[i] = v * sigmoid(v);
        }
        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_mul(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        let a = self.resolve_f32(graph, op.inputs[0])?;
        let b = self.resolve_f32(graph, op.inputs[1])?;
        let mut out = vec![0.0f32; a.len().min(b.len())];
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_add(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        let a = self.resolve_f32(graph, op.inputs[0])?;
        let b = self.resolve_f32(graph, op.inputs[1])?;
        let mut out = vec![0.0f32; a.len().max(b.len())];
        for i in 0..out.len() {
            out[i] = a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0);
        }
        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_softmax(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        let x = self.resolve_f32(graph, op.inputs[0])?;
        let n = x.len();
        let mut out = vec![0.0f32; n];

        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = x.iter().map(|&v| (v - max).exp()).sum();
        for (i, &v) in x.iter().enumerate() {
            out[i] = (v - max).exp() / sum;
        }
        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_embedding(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        // W: [vocab, d], ids: [batch]
        let w = self.resolve_f32(graph, op.inputs[0])?;
        let ids = self.resolve_f32(graph, op.inputs[1])?;
        let d = graph.tensors[op.inputs[0]].shape.dims[0]; // inner dim
        let batch = graph.tensors[op.inputs[1]].shape.dims[0];

        let mut out = vec![0.0f32; batch * d];
        for b in 0..batch {
            let id = ids[b] as usize;
            let src = &w[id * d..(id + 1) * d];
            out[b * d..(b + 1) * d].copy_from_slice(src);
        }
        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_argmax(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        let x = self.resolve_f32(graph, op.inputs[0])?;
        let (idx, _) = x
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        let out = vec![idx as f32];
        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_reshape(&mut self, _graph: &CGraph, _op: &Op) -> Result<()> {
        // Reshape is a no-op — the shape is already set on the output tensor.
        Ok(())
    }

    fn exec_transpose(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        let x = self.resolve_f32(graph, op.inputs[0])?;
        let shape = &graph.tensors[op.inputs[0]].shape;
        let rows = shape.dims[0];
        let cols = shape.dims[1];

        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = x[r * cols + c];
            }
        }
        self.scratch[op.output] = out;
        Ok(())
    }

    fn exec_swiglu(&mut self, graph: &CGraph, op: &Op) -> Result<()> {
        // SwiGLU: gate = SiLU(x @ W_gate), up = x @ W_up, y = gate * up @ W_down
        // We need 3 inputs: gate_proj_result, up_proj_result, down_weight
        // For now simplified: gate = SiLU(x_gate), then gate * x_up
        let gate = self.resolve_f32(graph, op.inputs[0])?;
        let up = self.resolve_f32(graph, op.inputs[1])?;
        let n = gate.len().min(up.len());
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = sigmoid(gate[i]) * gate[i] * up[i];
        }
        self.scratch[op.output] = out;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scalar math
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
