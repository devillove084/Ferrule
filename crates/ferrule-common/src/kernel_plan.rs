//! Kernel provider boundary and executable plan ABI.
//!
//! Implements Section 3.3 of the roadmap: multiple leaf kernel providers behind
//! a versioned POD descriptor, without hot-path dynamic trait dispatch.
//!
//! These types live in `ferrule-common` so they are available to both the CUDA
//! and CPU/reference execution paths.  The CUDA-specific provider
//! implementation (loading cubins, managing `LoadedModule`) lives in
//! `ferrule-cuda::provider`.
//!
//! Provider selection occurs during prepare/compile, not through string lookup
//! or hot-path dynamic policy.  The hot path reads pre-resolved POD
//! descriptors and dispatches directly.

use crate::Result;

// ── Row bucket ────────────────────────────────────────────────────────

/// Verification row bucket.  Maps directly to the roadmap's shape strategy
/// (Section 3.4): rows=1 is the persistent single-token path, rows=2/4 are
/// Tensor Core verification schedules, and rows=8 is enabled only when
/// acceptance/serving evidence justifies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RowBucket {
    /// Single-token decode.  Persistent small-M kernels; no padded batch-16 work.
    R1 = 1,
    /// 2-row verification batch.
    R2 = 2,
    /// 4-row verification batch.
    R4 = 4,
    /// 8-row batch (enabled only when acceptance evidence justifies it).
    R8 = 8,
}

impl RowBucket {
    /// Returns the row count for this bucket.
    pub const fn rows(self) -> usize {
        self as usize
    }

    /// Parses a row count into a bucket, or `None` if the count is not a
    /// supported bucket size.
    pub const fn from_rows(rows: usize) -> Option<Self> {
        match rows {
            1 => Some(Self::R1),
            2 => Some(Self::R2),
            4 => Some(Self::R4),
            8 => Some(Self::R8),
            _ => None,
        }
    }

    /// All buckets in escalation order.
    pub const ALL: [Self; 4] = [Self::R1, Self::R2, Self::R4, Self::R8];
}

// ── Provider and kernel identity ──────────────────────────────────────

/// Identifies which kernel provider owns a kernel launch.
///
/// This is a stable enum (not a string or trait object) so the hot path can
/// match on it without dynamic dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KernelProviderId {
    /// cuda-oxide compiled Rust kernels (`#[cuda_module]`).  Provider zero.
    /// Used for routing, metadata, recurrent compressor logic, paged control,
    /// custom HC glue, and eager correctness paths.
    CudaOxide = 0,
    /// Fixed-shape production cubins loaded through the CUDA Driver API.
    EmbeddedCubin = 1,
    /// CUTLASS/CuTe-generated cubins where they win.
    CutlassCubin = 2,
}

/// Identifies a specific kernel phase within a layer execution.
///
/// These correspond to the superkernel bundles in Section 3.5-3.8 of the
/// roadmap.  Each phase maps to one or more device kernel launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KernelPhase {
    /// Embedding lookup.
    Embed = 0,
    /// HC-pre + layer RMSNorm + FP8 pack (Bundle A).
    HcPre = 1,
    /// MLA query/KV projection (Bundle B, input side).
    MlaProjection = 2,
    /// Sparse attention.
    SparseAttention = 3,
    /// MLA output-A/output-B (Bundle B, output side).
    MlaOutput = 4,
    /// Shared FFN gate/up/down (Bundle C).
    SharedFfn = 5,
    /// Routed MoE gate/up FP4 (Bundle D).
    MoeGateUp = 6,
    /// Routed MoE down + rank-ordered reduction (Bundle D).
    MoeDown = 7,
    /// BF16 compressor dual projection.
    CompressorProjection = 8,
    /// FP8 activation pack.
    Fp8ActivationPack = 9,
    /// Output head: HC head + norm.
    OutputHeadNorm = 10,
    /// Output head: vocabulary projection / drafted-token verification.
    OutputHeadVocab = 11,
    /// Router scoring and top-k selection.
    Router = 12,
}

/// Unique identifier for a kernel within a provider's catalog.
///
/// The `provider` + `phase` + `rows` triple uniquely determines which kernel
/// function and launch configuration to use.  The `variant` field allows
/// multiple implementations of the same phase (e.g., a fast path and a
/// fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId {
    pub provider: KernelProviderId,
    pub phase: KernelPhase,
    pub rows: RowBucket,
    /// Implementation variant within the same provider/phase/rows.
    /// 0 = default/primary, 1+ = alternate implementations.
    pub variant: u8,
}

impl KernelId {
    pub const fn new(provider: KernelProviderId, phase: KernelPhase, rows: RowBucket) -> Self {
        Self {
            provider,
            phase,
            rows,
            variant: 0,
        }
    }

    pub const fn with_variant(mut self, variant: u8) -> Self {
        self.variant = variant;
        self
    }
}

// ── Launch descriptor ─────────────────────────────────────────────────

/// Versioned POD launch descriptor.
///
/// This structure describes how to launch a specific kernel without requiring
/// a trait object on the hot path.  The provider reads this descriptor and
/// dispatches to its internal launch method.
///
/// Future versions may add persistent tile schedules, shared-memory sizes,
/// and register allocation hints.  The `version` field allows safe evolution.
#[derive(Debug, Clone, Copy)]
pub struct LaunchDescriptor {
    /// ABI version of this descriptor.  Must match what the provider expects.
    pub version: u16,
    /// Which kernel to launch.
    pub kernel: KernelId,
    /// Grid dimensions (x, y, z).
    pub grid: (u32, u32, u32),
    /// Block dimensions (x, y, z).
    pub block: (u32, u32, u32),
    /// Shared memory per block in bytes.
    pub shared_mem_bytes: u32,
    /// Whether this kernel is safe inside CUDA graph capture (no D2H, no
    /// allocation, no stream-wide sync).
    pub capture_safe: bool,
}

impl LaunchDescriptor {
    pub const ABI_VERSION: u16 = 1;

    pub const fn new(kernel: KernelId, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Self {
        Self {
            version: Self::ABI_VERSION,
            kernel,
            grid,
            block,
            shared_mem_bytes: 0,
            capture_safe: false,
        }
    }

    pub const fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    pub const fn capture_safe(mut self) -> Self {
        self.capture_safe = true;
        self
    }
}

// ── Weight layout descriptor ──────────────────────────────────────────

/// Backend-native weight layout produced during ingest.
///
/// Section 3.3: "provider-native weight transforms are performed once during
/// ingest into the final unique device layout."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WeightLayout {
    /// Original row-major layout (cuda-oxide default).
    RowMajor = 0,
    /// Transposed for device HC accumulation order.
    TransposedRowMajor = 1,
    /// FP4 packed with E8M0 scale, expert-major.
    Fp4PackedExpertMajor = 2,
    /// FP8 E4M3 with E8M0 block scale.
    Fp8E4m3BlockScaled = 3,
    /// BF16 row-major.
    Bf16RowMajor = 4,
    /// CUTLASS-style interleaved layout (provider-specific).
    CutlassInterleaved = 5,
}

/// Describes a weight tensor's device layout and workspace offset.
#[derive(Debug, Clone, Copy)]
pub struct WeightBinding {
    /// Device-native layout of this weight tensor.
    pub layout: WeightLayout,
    /// Byte offset into the layer's persistent weight arena.
    pub offset: u64,
    /// Byte length of this weight tensor in device memory.
    pub len: u64,
}

// ── Layer kernel plan ─────────────────────────────────────────────────

/// Per-shape kernel plan for one layer.
///
/// Section 3.2: each layer carries a `LayerKernelPlan[rows=1/2/4/8]` that
/// binds specific kernel IDs, weight layouts, tile schedules, fusion
/// descriptors, workspace offsets, and graph bucket bindings.
///
/// This structure is the compiled output of prepare/compile, not a hot-path
/// decision.  The execution loop reads the plan for the active row bucket and
/// dispatches directly.
#[derive(Debug, Clone)]
pub struct LayerKernelPlan {
    /// Row bucket this plan is compiled for.
    pub rows: RowBucket,
    /// Launch descriptor for each kernel phase in this layer.
    pub phases: Vec<LaunchDescriptor>,
    /// Weight bindings for this layer's tensors, indexed by phase.
    pub weights: Vec<WeightBinding>,
    /// Workspace byte offset within the persistent arena.
    pub workspace_offset: u64,
    /// Workspace byte length for this plan.
    pub workspace_len: u64,
    /// Whether this plan is fully resident (no I/O) and capture-safe.
    pub resident_capture_safe: bool,
}

impl LayerKernelPlan {
    /// Creates an empty plan for the given row bucket.
    pub fn new(rows: RowBucket) -> Self {
        Self {
            rows,
            phases: Vec::new(),
            weights: Vec::new(),
            workspace_offset: 0,
            workspace_len: 0,
            resident_capture_safe: false,
        }
    }

    /// Returns the launch descriptor for a specific phase, if present.
    pub fn phase(&self, phase: KernelPhase) -> Option<&LaunchDescriptor> {
        self.phases.iter().find(|d| d.kernel.phase == phase)
    }

    /// Adds or replaces a phase launch descriptor.
    pub fn set_phase(&mut self, descriptor: LaunchDescriptor) {
        if let Some(existing) = self
            .phases
            .iter_mut()
            .find(|d| d.kernel.phase == descriptor.kernel.phase)
        {
            *existing = descriptor;
        } else {
            self.phases.push(descriptor);
        }
    }

    /// Marks this plan as fully resident and capture-safe.
    pub fn mark_resident_capture_safe(&mut self) {
        self.resident_capture_safe = true;
    }
}

/// Plans for all row buckets of one layer.
///
/// Indexed by [`RowBucket`].  Not all buckets need plans; unset buckets
/// fall back to the eager path.
#[derive(Debug, Clone, Default)]
pub struct LayerKernelPlanSet {
    r1: Option<LayerKernelPlan>,
    r2: Option<LayerKernelPlan>,
    r4: Option<LayerKernelPlan>,
    r8: Option<LayerKernelPlan>,
}

impl LayerKernelPlanSet {
    /// Creates an empty plan set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the plan for the given row bucket, if compiled.
    pub fn plan(&self, rows: RowBucket) -> Option<&LayerKernelPlan> {
        match rows {
            RowBucket::R1 => self.r1.as_ref(),
            RowBucket::R2 => self.r2.as_ref(),
            RowBucket::R4 => self.r4.as_ref(),
            RowBucket::R8 => self.r8.as_ref(),
        }
    }

    /// Returns a mutable reference to the plan for the given row bucket.
    pub fn plan_mut(&mut self, rows: RowBucket) -> &mut Option<LayerKernelPlan> {
        match rows {
            RowBucket::R1 => &mut self.r1,
            RowBucket::R2 => &mut self.r2,
            RowBucket::R4 => &mut self.r4,
            RowBucket::R8 => &mut self.r8,
        }
    }

    /// Sets the plan for the given row bucket.
    pub fn set_plan(&mut self, plan: LayerKernelPlan) {
        let slot = self.plan_mut(plan.rows);
        *slot = Some(plan);
    }

    /// Returns `true` if any plan is compiled for the given row bucket.
    pub fn has_plan(&self, rows: RowBucket) -> bool {
        self.plan(rows).is_some()
    }

    /// Returns `true` if the plan for the given rows is marked
    /// resident-capture-safe.
    pub fn is_resident_capture_safe(&self, rows: RowBucket) -> bool {
        self.plan(rows)
            .is_some_and(|plan| plan.resident_capture_safe)
    }
}

// ── Model kernel plan ─────────────────────────────────────────────────

/// Kernel plans for all layers of a model.
///
/// This is the "executable plan" side of the execution image (Section 3.2).
/// It is compiled during `prepare` and stored alongside the resource image.
#[derive(Debug, Clone)]
pub struct ModelKernelPlan {
    /// Per-layer plan sets, indexed by layer number.
    pub layers: Vec<LayerKernelPlanSet>,
}

impl ModelKernelPlan {
    /// Creates an empty model plan with the given layer count.
    pub fn new(layer_count: usize) -> Self {
        Self {
            layers: vec![LayerKernelPlanSet::new(); layer_count],
        }
    }

    /// Returns the plan set for the given layer.
    pub fn layer(&self, layer: usize) -> Option<&LayerKernelPlanSet> {
        self.layers.get(layer)
    }

    /// Returns a mutable reference to the plan set for the given layer.
    pub fn layer_mut(&mut self, layer: usize) -> Option<&mut LayerKernelPlanSet> {
        self.layers.get_mut(layer)
    }

    /// Returns `true` if any layer has a compiled plan for the given rows.
    pub fn has_any_plan(&self, rows: RowBucket) -> bool {
        self.layers.iter().any(|set| set.has_plan(rows))
    }
}

// ── Provider manifest ─────────────────────────────────────────────────

/// Static metadata about a kernel provider, registered at startup.
#[derive(Debug, Clone)]
pub struct ProviderManifest {
    pub id: KernelProviderId,
    pub name: &'static str,
    pub abi_version: u16,
    /// Number of kernel variants registered by this provider.
    pub kernel_count: usize,
}

impl ProviderManifest {
    pub const fn cuda_oxide() -> Self {
        Self {
            id: KernelProviderId::CudaOxide,
            name: "cuda-oxide",
            abi_version: LaunchDescriptor::ABI_VERSION,
            kernel_count: 0,
        }
    }

    pub const fn embedded_cubin() -> Self {
        Self {
            id: KernelProviderId::EmbeddedCubin,
            name: "embedded-cubin",
            abi_version: LaunchDescriptor::ABI_VERSION,
            kernel_count: 0,
        }
    }
}

/// Registry of available kernel providers.
///
/// Built once during context creation.  The hot path does not query this;
/// it reads pre-resolved `KernelId` values from the `ModelKernelPlan`.
#[derive(Debug, Clone, Default)]
pub struct ProviderRegistry {
    providers: Vec<ProviderManifest>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers the cuda-oxide provider as provider zero.
    pub fn with_cuda_oxide() -> Self {
        let mut registry = Self::new();
        registry.register(ProviderManifest::cuda_oxide());
        registry
    }

    pub fn register(&mut self, manifest: ProviderManifest) {
        self.providers.push(manifest);
    }

    pub fn is_available(&self, id: KernelProviderId) -> bool {
        self.providers.iter().any(|m| m.id == id)
    }

    pub fn manifests(&self) -> &[ProviderManifest] {
        &self.providers
    }
}

// ── Default plan compilation ──────────────────────────────────────────

/// Compiles a default model kernel plan for the given layer count.
///
/// This creates a plan where every layer uses the cuda-oxide provider (the
/// existing eager path) with no optimized schedules.  As superkernel bundles
/// are implemented, individual phase descriptors are filled in or replaced.
///
/// The plan starts empty (no compiled phases) and phases are added as
/// kernels are registered.  An empty plan means "use the existing eager
/// dispatch path" - the runner checks `has_any_plan` before consulting the
/// plan.
pub fn compile_default_plan(layer_count: usize) -> Result<ModelKernelPlan> {
    Ok(ModelKernelPlan::new(layer_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bucket_from_rows_roundtrips() {
        assert_eq!(RowBucket::from_rows(1), Some(RowBucket::R1));
        assert_eq!(RowBucket::from_rows(2), Some(RowBucket::R2));
        assert_eq!(RowBucket::from_rows(4), Some(RowBucket::R4));
        assert_eq!(RowBucket::from_rows(8), Some(RowBucket::R8));
        assert_eq!(RowBucket::from_rows(3), None);
        assert_eq!(RowBucket::from_rows(16), None);
    }

    #[test]
    fn row_bucket_rows() {
        assert_eq!(RowBucket::R1.rows(), 1);
        assert_eq!(RowBucket::R2.rows(), 2);
        assert_eq!(RowBucket::R4.rows(), 4);
        assert_eq!(RowBucket::R8.rows(), 8);
    }

    #[test]
    fn kernel_id_construction() {
        let id = KernelId::new(
            KernelProviderId::CudaOxide,
            KernelPhase::HcPre,
            RowBucket::R1,
        );
        assert_eq!(id.provider, KernelProviderId::CudaOxide);
        assert_eq!(id.phase, KernelPhase::HcPre);
        assert_eq!(id.rows, RowBucket::R1);
        assert_eq!(id.variant, 0);

        let id_v2 = id.with_variant(1);
        assert_eq!(id_v2.variant, 1);
        assert_eq!(id.variant, 0); // original unchanged
    }

    #[test]
    fn launch_descriptor_versioning() {
        let id = KernelId::new(
            KernelProviderId::CudaOxide,
            KernelPhase::Embed,
            RowBucket::R1,
        );
        let desc = LaunchDescriptor::new(id, (1, 1, 1), (256, 1, 1));
        assert_eq!(desc.version, LaunchDescriptor::ABI_VERSION);
        assert!(!desc.capture_safe);

        let desc_safe = desc.capture_safe().with_shared_mem(4096);
        assert!(desc_safe.capture_safe);
        assert_eq!(desc_safe.shared_mem_bytes, 4096);
    }

    #[test]
    fn layer_kernel_plan_set_phase() {
        let mut plan = LayerKernelPlan::new(RowBucket::R1);
        let id = KernelId::new(
            KernelProviderId::CudaOxide,
            KernelPhase::HcPre,
            RowBucket::R1,
        );
        let desc = LaunchDescriptor::new(id, (1, 1, 1), (128, 1, 1));

        assert!(plan.phase(KernelPhase::HcPre).is_none());
        plan.set_phase(desc);
        assert!(plan.phase(KernelPhase::HcPre).is_some());
        assert!(plan.phase(KernelPhase::Embed).is_none());

        // Replace existing phase
        let desc2 = LaunchDescriptor::new(id, (2, 1, 1), (128, 1, 1));
        plan.set_phase(desc2);
        assert_eq!(plan.phase(KernelPhase::HcPre).unwrap().grid.0, 2);
    }

    #[test]
    fn layer_kernel_plan_set_buckets() {
        let mut set = LayerKernelPlanSet::new();
        assert!(!set.has_plan(RowBucket::R1));
        assert!(!set.has_plan(RowBucket::R4));

        let plan_r1 = LayerKernelPlan::new(RowBucket::R1);
        set.set_plan(plan_r1);
        assert!(set.has_plan(RowBucket::R1));
        assert!(!set.has_plan(RowBucket::R4));

        let mut plan_r4 = LayerKernelPlan::new(RowBucket::R4);
        plan_r4.mark_resident_capture_safe();
        set.set_plan(plan_r4);
        assert!(set.has_plan(RowBucket::R4));
        assert!(set.is_resident_capture_safe(RowBucket::R4));
        assert!(!set.is_resident_capture_safe(RowBucket::R1));
    }

    #[test]
    fn model_kernel_plan_layer_access() {
        let mut plan = compile_default_plan(43).unwrap();
        assert_eq!(plan.layers.len(), 43);
        assert!(plan.layer(0).is_some());
        assert!(plan.layer(43).is_none());

        let plan_r1 = LayerKernelPlan::new(RowBucket::R1);
        plan.layer_mut(0).unwrap().set_plan(plan_r1);
        assert!(plan.has_any_plan(RowBucket::R1));
        assert!(!plan.has_any_plan(RowBucket::R2));
    }

    #[test]
    fn compile_default_plan_allows_zero_layers() {
        let plan = compile_default_plan(0).unwrap();
        assert!(plan.layers.is_empty());
    }

    #[test]
    fn provider_registry_cuda_oxide() {
        let registry = ProviderRegistry::with_cuda_oxide();
        assert!(registry.is_available(KernelProviderId::CudaOxide));
        assert!(!registry.is_available(KernelProviderId::EmbeddedCubin));
        assert_eq!(registry.manifests().len(), 1);
        assert_eq!(registry.manifests()[0].name, "cuda-oxide");
    }

    #[test]
    fn weight_layout_repr() {
        let layout = WeightLayout::RowMajor;
        assert_eq!(layout as u8, 0);
        let layout = WeightLayout::Fp4PackedExpertMajor;
        assert_eq!(layout as u8, 2);
    }

    #[test]
    fn all_row_buckets_in_order() {
        assert_eq!(
            RowBucket::ALL,
            [RowBucket::R1, RowBucket::R2, RowBucket::R4, RowBucket::R8]
        );
    }
}
