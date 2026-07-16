//! Kernel provider boundary and executable plan ABI.
//!
//! Implements Section 3.3 of the roadmap: multiple leaf kernel providers behind
//! a versioned POD descriptor, without hot-path dynamic trait dispatch.
//!
//! These provider-neutral types live in `ferrule-common`; the GB10 CUDA
//! provider implementation lives in `ferrule-cuda::provider`.
//!
//! Provider selection occurs during prepare/compile, not through string lookup
//! or hot-path dynamic policy.  The hot path reads pre-resolved POD
//! descriptors and dispatches directly.

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
    /// Stable-frame routed FP4 expert bundle (Bundle D).
    RoutedMoe = 6,
    /// BF16 compressor dual projection.
    CompressorProjection = 7,
    /// FP8 activation pack.
    Fp8ActivationPack = 8,
    /// Output head: HC head + norm.
    OutputHeadNorm = 9,
    /// Output head: vocabulary projection / drafted-token verification.
    OutputHeadVocab = 10,
    /// Router scoring and top-k selection.
    Router = 11,
}

/// Stable semantic operation bound to a provider kernel.
///
/// A phase is only a profiling/scheduling group and is not a unique binding
/// key: MLA projection, for example, contains QueryA, QueryB, KV, and indexer
/// consumers. The operation is therefore stored in the executable plan while
/// [`KernelOperation::phase`] recovers the coarse group when needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KernelOperation {
    Embed = 0,
    AttentionHcPre = 1,
    FeedForwardHcPre = 2,
    MlaQueryA = 3,
    MlaQueryB = 4,
    MlaKeyValue = 5,
    SparseAttention = 6,
    /// One-launch grouped output-A -> BF16 latent -> output-B bundle.
    MlaOutput = 7,
    /// One-launch shared gate/up -> SwiGLU -> down bundle.
    SharedFfn = 9,
    /// One-launch stable-frame routed gate/up -> SwiGLU -> down bundle.
    RoutedFp4Moe = 10,
    MainCompressorProjection = 11,
    IndexerCompressorProjection = 12,
    Fp8ActivationPack = 13,
    OutputHeadNorm = 14,
    OutputHeadVocab = 15,
    Router = 16,
    IndexerQuery = 17,
    IndexerWeights = 18,
    /// One-launch QueryA + KV multi-output projection bundle.
    MlaQueryAKv = 19,
}

impl KernelOperation {
    pub const fn phase(self) -> KernelPhase {
        match self {
            Self::Embed => KernelPhase::Embed,
            Self::AttentionHcPre | Self::FeedForwardHcPre => KernelPhase::HcPre,
            Self::MlaQueryA
            | Self::MlaQueryB
            | Self::MlaKeyValue
            | Self::MlaQueryAKv
            | Self::IndexerQuery
            | Self::IndexerWeights => KernelPhase::MlaProjection,
            Self::SparseAttention => KernelPhase::SparseAttention,
            Self::MlaOutput => KernelPhase::MlaOutput,
            Self::SharedFfn => KernelPhase::SharedFfn,
            Self::RoutedFp4Moe => KernelPhase::RoutedMoe,
            Self::MainCompressorProjection | Self::IndexerCompressorProjection => {
                KernelPhase::CompressorProjection
            }
            Self::Fp8ActivationPack => KernelPhase::Fp8ActivationPack,
            Self::OutputHeadNorm => KernelPhase::OutputHeadNorm,
            Self::OutputHeadVocab => KernelPhase::OutputHeadVocab,
            Self::Router => KernelPhase::Router,
        }
    }
}

/// Unique identifier for a kernel within a provider's catalog.
///
/// The `provider` + `operation` pair uniquely determines the semantic binding.
/// Architecture-specific M ranges and tile schedules stay inside the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct KernelId {
    pub provider: KernelProviderId,
    pub operation: KernelOperation,
    /// Semantic implementation revision, never a runtime M bucket.
    pub variant: u8,
    /// Explicit padding reserved for ABI evolution.
    pub reserved: u8,
}

impl KernelId {
    pub const fn new(provider: KernelProviderId, operation: KernelOperation) -> Self {
        Self {
            provider,
            operation,
            variant: 0,
            reserved: 0,
        }
    }

    pub const fn phase(self) -> KernelPhase {
        self.operation.phase()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct LaunchDescriptor {
    /// ABI version of this descriptor. Must match what the provider expects.
    pub version: u16,
    /// Stable bit flags; unlike Rust `bool`, these have an explicit C ABI.
    pub flags: u16,
    /// Which kernel to launch.
    pub kernel: KernelId,
    /// Grid dimensions (x, y, z). Zeroes are valid only for provider-managed
    /// launches whose concrete geometry is resolved from the POD arguments.
    pub grid: [u32; 3],
    /// Block dimensions (x, y, z).
    pub block: [u32; 3],
    /// Shared memory per block in bytes.
    pub shared_mem_bytes: u32,
    /// Reserved for ABI-compatible extension and explicit 8-byte tail size.
    pub reserved: u32,
}

impl LaunchDescriptor {
    pub const ABI_VERSION: u16 = 2;
    pub const FLAG_CAPTURE_SAFE: u16 = 1 << 0;
    pub const FLAG_PROVIDER_MANAGED_LAUNCH: u16 = 1 << 1;

    pub const fn new(kernel: KernelId, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Self {
        Self {
            version: Self::ABI_VERSION,
            flags: 0,
            kernel,
            grid: [grid.0, grid.1, grid.2],
            block: [block.0, block.1, block.2],
            shared_mem_bytes: 0,
            reserved: 0,
        }
    }

    pub const fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    pub const fn capture_safe(mut self) -> Self {
        self.flags |= Self::FLAG_CAPTURE_SAFE;
        self
    }

    pub const fn provider_managed(mut self) -> Self {
        self.flags |= Self::FLAG_PROVIDER_MANAGED_LAUNCH;
        self
    }

    pub const fn is_capture_safe(self) -> bool {
        self.flags & Self::FLAG_CAPTURE_SAFE != 0
    }

    pub const fn is_provider_managed(self) -> bool {
        self.flags & Self::FLAG_PROVIDER_MANAGED_LAUNCH != 0
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

/// Provider-neutral requirement for one linear bundle.
///
/// Multiple outputs share one input activation and producer contract. For
/// example, compressor KV/gate are represented as one semantic operation with
/// two output widths, allowing a provider to share the activation producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearBundleRequirement {
    pub operation: KernelOperation,
    pub input_features: usize,
    pub output_features: Box<[usize]>,
    pub weight_layout: WeightLayout,
}

impl LinearBundleRequirement {
    pub fn new(
        operation: KernelOperation,
        input_features: usize,
        output_features: impl Into<Box<[usize]>>,
        weight_layout: WeightLayout,
    ) -> Self {
        Self {
            operation,
            input_features,
            output_features: output_features.into(),
            weight_layout,
        }
    }
}

/// Provider-neutral operation requirements for one model layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerKernelRequirements {
    pub linear_bundles: Vec<LinearBundleRequirement>,
    pub semantic_operations: Vec<KernelOperation>,
}

impl LayerKernelRequirements {
    pub fn add_linear_bundle(&mut self, requirement: LinearBundleRequirement) {
        self.linear_bundles.push(requirement);
    }

    pub fn require_operation(&mut self, operation: KernelOperation) {
        if !self.semantic_operations.contains(&operation) {
            self.semantic_operations.push(operation);
        }
    }
}

/// Describes a weight tensor's semantic operand and device layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct WeightBinding {
    /// Operation that consumes this tensor.
    pub operation: KernelOperation,
    /// Device-native layout of this weight tensor.
    pub layout: WeightLayout,
    /// Operand index within the operation (for example KV=0, gate=1).
    pub operand: u8,
    /// Explicit padding reserved for ABI-compatible flags.
    pub reserved: [u8; 5],
    /// Byte offset into the layer's persistent weight arena.
    pub offset: u64,
    /// Byte length of this weight tensor in device memory.
    pub len: u64,
}

impl WeightBinding {
    pub const fn new(
        operation: KernelOperation,
        operand: u8,
        layout: WeightLayout,
        offset: u64,
        len: u64,
    ) -> Self {
        Self {
            operation,
            layout,
            operand,
            reserved: [0; 5],
            offset,
            len,
        }
    }
}

// ── Layer kernel plan ─────────────────────────────────────────────────

/// Semantic kernel plan for one layer.
///
/// Row-count schedule dispatch is provider-owned. CUDA Graph capture buckets are
/// a separate runtime concern and must not duplicate semantic bindings.
#[derive(Debug, Clone)]
pub struct LayerKernelPlan {
    /// Launch descriptor for each semantic operation in this layer.
    pub launches: Vec<LaunchDescriptor>,
    /// Weight bindings keyed by semantic operation and operand.
    pub weights: Vec<WeightBinding>,
    /// Workspace byte offset within the persistent arena.
    pub workspace_offset: u64,
    /// Workspace byte length for this plan.
    pub workspace_len: u64,
    /// Whether this plan is fully resident (no I/O) and capture-safe.
    pub resident_capture_safe: bool,
}

impl LayerKernelPlan {
    /// Creates an empty semantic plan.
    pub fn new() -> Self {
        Self {
            launches: Vec::new(),
            weights: Vec::new(),
            workspace_offset: 0,
            workspace_len: 0,
            resident_capture_safe: false,
        }
    }

    /// Returns the launch descriptor for one exact semantic operation.
    pub fn operation(&self, operation: KernelOperation) -> Option<&LaunchDescriptor> {
        self.launches
            .iter()
            .find(|descriptor| descriptor.kernel.operation == operation)
    }

    /// Iterates every operation in a coarse profiling/scheduling phase.
    pub fn operations_in_phase(
        &self,
        phase: KernelPhase,
    ) -> impl Iterator<Item = &LaunchDescriptor> {
        self.launches
            .iter()
            .filter(move |descriptor| descriptor.kernel.phase() == phase)
    }

    /// Adds or replaces one exact semantic-operation launch descriptor.
    pub fn set_operation(&mut self, descriptor: LaunchDescriptor) {
        if let Some(existing) = self
            .launches
            .iter_mut()
            .find(|existing| existing.kernel.operation == descriptor.kernel.operation)
        {
            *existing = descriptor;
        } else {
            self.launches.push(descriptor);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.launches.is_empty()
    }

    /// Marks this plan as fully resident and capture-safe.
    pub fn mark_resident_capture_safe(&mut self) {
        self.resident_capture_safe = true;
    }
}

// ── Model kernel plan ─────────────────────────────────────────────────

/// Kernel plans for all layers of a model.
///
/// This is the "executable plan" side of the execution image (Section 3.2).
/// It is compiled during `prepare` and stored alongside the resource image.
#[derive(Debug, Clone)]
pub struct ModelKernelPlan {
    /// Per-layer semantic plans, indexed by layer number.
    pub layers: Vec<LayerKernelPlan>,
}

impl ModelKernelPlan {
    /// Creates an empty model plan with the given layer count.
    pub fn new(layer_count: usize) -> Self {
        Self {
            layers: (0..layer_count).map(|_| LayerKernelPlan::new()).collect(),
        }
    }

    /// Returns the plan set for the given layer.
    pub fn layer(&self, layer: usize) -> Option<&LayerKernelPlan> {
        self.layers.get(layer)
    }

    /// Returns a mutable reference to one layer's semantic plan.
    pub fn layer_mut(&mut self, layer: usize) -> Option<&mut LayerKernelPlan> {
        self.layers.get_mut(layer)
    }

    pub fn has_operation(&self, layer: usize, operation: KernelOperation) -> bool {
        self.layer(layer)
            .and_then(|plan| plan.operation(operation))
            .is_some()
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

    /// Manifest for a CUTLASS-generated cubin/native provider. The provider's
    /// own POD ABI version is recorded independently from the execution-plan
    /// launch descriptor version.
    pub const fn cutlass_cubin(abi_version: u16, kernel_count: usize) -> Self {
        Self {
            id: KernelProviderId::CutlassCubin,
            name: "cutlass-cubin",
            abi_version,
            kernel_count,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_id_construction() {
        let id = KernelId::new(KernelProviderId::CudaOxide, KernelOperation::AttentionHcPre);
        assert_eq!(id.provider, KernelProviderId::CudaOxide);
        assert_eq!(id.operation, KernelOperation::AttentionHcPre);
        assert_eq!(id.phase(), KernelPhase::HcPre);
        assert_eq!(id.variant, 0);
        assert_eq!(id.reserved, 0);

        let id_v2 = id.with_variant(1);
        assert_eq!(id_v2.variant, 1);
        assert_eq!(id.variant, 0); // original unchanged
    }

    #[test]
    fn launch_descriptor_has_stable_pod_layout() {
        assert_eq!(std::mem::size_of::<KernelId>(), 4);
        assert_eq!(std::mem::align_of::<KernelId>(), 1);
        assert_eq!(std::mem::size_of::<LaunchDescriptor>(), 40);
        assert_eq!(std::mem::align_of::<LaunchDescriptor>(), 4);
        assert_eq!(std::mem::size_of::<WeightBinding>(), 24);
        assert_eq!(std::mem::align_of::<WeightBinding>(), 8);
    }

    #[test]
    fn launch_descriptor_versioning() {
        let id = KernelId::new(KernelProviderId::CudaOxide, KernelOperation::Embed);
        let desc = LaunchDescriptor::new(id, (1, 1, 1), (256, 1, 1));
        assert_eq!(desc.version, LaunchDescriptor::ABI_VERSION);
        assert!(!desc.is_capture_safe());

        let desc_safe = desc.capture_safe().with_shared_mem(4096);
        assert!(desc_safe.is_capture_safe());
        assert_eq!(desc_safe.shared_mem_bytes, 4096);
    }

    #[test]
    fn layer_kernel_plan_binds_operations_not_phases() {
        let mut plan = LayerKernelPlan::new();
        let attention_id =
            KernelId::new(KernelProviderId::CudaOxide, KernelOperation::AttentionHcPre);
        let feed_forward_id = KernelId::new(
            KernelProviderId::CudaOxide,
            KernelOperation::FeedForwardHcPre,
        );
        plan.set_operation(LaunchDescriptor::new(attention_id, (1, 1, 1), (128, 1, 1)));
        plan.set_operation(LaunchDescriptor::new(
            feed_forward_id,
            (2, 1, 1),
            (128, 1, 1),
        ));

        assert_eq!(plan.operations_in_phase(KernelPhase::HcPre).count(), 2);
        assert_eq!(
            plan.operation(KernelOperation::AttentionHcPre)
                .unwrap()
                .grid[0],
            1
        );

        // Replace only the exact operation, not another launch in the phase.
        plan.set_operation(LaunchDescriptor::new(attention_id, (3, 1, 1), (128, 1, 1)));
        assert_eq!(plan.operations_in_phase(KernelPhase::HcPre).count(), 2);
        assert_eq!(
            plan.operation(KernelOperation::AttentionHcPre)
                .unwrap()
                .grid[0],
            3
        );
        assert_eq!(
            plan.operation(KernelOperation::FeedForwardHcPre)
                .unwrap()
                .grid[0],
            2
        );
    }

    #[test]
    fn model_kernel_plan_layer_access() {
        let mut plan = ModelKernelPlan::new(43);
        assert_eq!(plan.layers.len(), 43);
        assert!(plan.layer(0).is_some());
        assert!(plan.layer(43).is_none());

        let launch = LaunchDescriptor::new(
            KernelId::new(KernelProviderId::CudaOxide, KernelOperation::Embed),
            (1, 1, 1),
            (128, 1, 1),
        );
        plan.layer_mut(0).unwrap().set_operation(launch);
        assert!(plan.has_operation(0, KernelOperation::Embed));
        assert!(!plan.has_operation(0, KernelOperation::Router));
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
    fn cutlass_manifest_keeps_provider_abi_and_catalog_size() {
        let manifest = ProviderManifest::cutlass_cubin(3, 2);
        assert_eq!(manifest.id, KernelProviderId::CutlassCubin);
        assert_eq!(manifest.abi_version, 3);
        assert_eq!(manifest.kernel_count, 2);
    }

    #[test]
    fn weight_layout_repr() {
        let layout = WeightLayout::RowMajor;
        assert_eq!(layout as u8, 0);
        let layout = WeightLayout::Fp4PackedExpertMajor;
        assert_eq!(layout as u8, 2);
    }
}
