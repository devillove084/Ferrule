//! Kernel provider boundary and executable plan contract.
//!
//! Implements Section 3.3 of the roadmap: multiple leaf kernel providers behind
//! a plain POD descriptor, without hot-path dynamic trait dispatch.
//!
//! These provider-neutral types live at the root of `ferrule-backend`; concrete
//! implementations such as NVIDIA CUDA remain in device-specific modules.
//!
//! Provider selection occurs during prepare/compile, not through string lookup
//! or hot-path dynamic policy.  The hot path reads pre-resolved POD
//! descriptors and dispatches directly.

// ── Provider and kernel identity ──────────────────────────────────────

/// Stable provider identity assigned during backend construction.
///
/// The identity deliberately says nothing about CUDA, CUTLASS, dynamic loading,
/// or code-object format. Device implementations register concrete providers
/// before model preparation; the hot path only reads this compact ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct KernelProviderId(u8);

impl KernelProviderId {
    pub const BUILTIN: Self = Self(0);
    pub const CUDA_CUTLASS: Self = Self(1);

    pub const fn new(value: u8) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
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
    /// Grouped FP4 expert bundle (Bundle D).
    GroupedMoe = 6,
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
    /// Proposal attachment operations.
    ProposalAttachment = 12,
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
    /// One-launch grouped gate/up -> SwiGLU -> down bundle.
    GroupedFp4Moe = 10,
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
    /// Proposal stage-zero target-tap projection followed by RMSNorm.
    MainProjectNorm = 20,
    /// Checkpoint-native proposal non-causal block attention over committed paged
    /// context plus ephemeral proposal KV.
    HybridMlaAttention = 21,
    /// Checkpoint-native proposal HC head, base LM projection, sequential Markov
    /// proposal selection, and confidence bundle.
    ProposalHead = 22,
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
            Self::GroupedFp4Moe => KernelPhase::GroupedMoe,
            Self::MainCompressorProjection | Self::IndexerCompressorProjection => {
                KernelPhase::CompressorProjection
            }
            Self::Fp8ActivationPack => KernelPhase::Fp8ActivationPack,
            Self::OutputHeadNorm => KernelPhase::OutputHeadNorm,
            Self::OutputHeadVocab => KernelPhase::OutputHeadVocab,
            Self::Router => KernelPhase::Router,
            Self::MainProjectNorm | Self::HybridMlaAttention | Self::ProposalHead => {
                KernelPhase::ProposalAttachment
            }
        }
    }
}

/// Execution role required from an operator implementation.
///
/// RL actors use `Inference`; learners use the same training modes as ordinary
/// supervised or post-training workloads. This keeps workload policy outside
/// the kernel-provider contract while making missing backward/update support
/// impossible to hide behind a forward-only binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExecutionMode {
    Inference = 0,
    TrainingForward = 1,
    Backward = 2,
    Optimizer = 3,
}

/// Compact set of execution modes published for one semantic operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ExecutionModeSet(u8);

impl ExecutionModeSet {
    pub const NONE: Self = Self(0);
    pub const INFERENCE: Self = Self(1 << ExecutionMode::Inference as u8);
    pub const TRAINING_FORWARD: Self = Self(1 << ExecutionMode::TrainingForward as u8);
    pub const BACKWARD: Self = Self(1 << ExecutionMode::Backward as u8);
    pub const OPTIMIZER: Self = Self(1 << ExecutionMode::Optimizer as u8);

    pub const fn contains(self, mode: ExecutionMode) -> bool {
        self.0 & (1 << mode as u8) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Unique semantic kernel binding within a provider catalog.
///
/// Architecture-specific M ranges and tile schedules stay inside the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct KernelId {
    pub provider: KernelProviderId,
    pub operation: KernelOperation,
    pub mode: ExecutionMode,
    /// Semantic implementation revision, never a runtime M bucket.
    pub variant: u8,
}

impl KernelId {
    pub const fn new(
        provider: KernelProviderId,
        operation: KernelOperation,
        mode: ExecutionMode,
    ) -> Self {
        Self {
            provider,
            operation,
            mode,
            variant: 0,
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

/// POD launch descriptor.
///
/// This structure describes how to launch a specific kernel without requiring
/// a trait object on the hot path. The provider reads this descriptor and
/// dispatches to its internal launch method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct LaunchDescriptor {
    /// Stable bit flags; unlike Rust `bool`, these have an explicit C ABI.
    pub flags: u16,
    /// Explicit padding keeps the following POD identity naturally aligned.
    pub reserved0: u16,
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
    pub const FLAG_CAPTURE_SAFE: u16 = 1 << 0;
    pub const FLAG_PROVIDER_MANAGED_LAUNCH: u16 = 1 << 1;

    pub const fn new(kernel: KernelId, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Self {
        Self {
            flags: 0,
            reserved0: 0,
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
    /// Original provider-neutral row-major layout.
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
    pub mode: ExecutionMode,
    pub input_features: usize,
    pub output_features: Box<[usize]>,
    pub weight_layout: WeightLayout,
}

impl LinearBundleRequirement {
    pub fn new(
        operation: KernelOperation,
        mode: ExecutionMode,
        input_features: usize,
        output_features: impl Into<Box<[usize]>>,
        weight_layout: WeightLayout,
    ) -> Self {
        Self {
            operation,
            mode,
            input_features,
            output_features: output_features.into(),
            weight_layout,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationRequirement {
    pub operation: KernelOperation,
    pub mode: ExecutionMode,
    pub deterministic: bool,
}

impl OperationRequirement {
    pub const fn new(operation: KernelOperation, mode: ExecutionMode) -> Self {
        Self {
            operation,
            mode,
            deterministic: false,
        }
    }

    pub const fn deterministic(mut self) -> Self {
        self.deterministic = true;
        self
    }
}

/// Provider-neutral operation requirements for one model layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerKernelRequirements {
    pub linear_bundles: Vec<LinearBundleRequirement>,
    pub operations: Vec<OperationRequirement>,
}

impl LayerKernelRequirements {
    pub fn add_linear_bundle(&mut self, requirement: LinearBundleRequirement) {
        self.linear_bundles.push(requirement);
    }

    pub fn require(&mut self, requirement: OperationRequirement) {
        if !self.operations.contains(&requirement) {
            self.operations.push(requirement);
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

impl Default for LayerKernelPlan {
    fn default() -> Self {
        Self::new()
    }
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
    pub fn operation(
        &self,
        operation: KernelOperation,
        mode: ExecutionMode,
    ) -> Option<&LaunchDescriptor> {
        self.launches.iter().find(|descriptor| {
            descriptor.kernel.operation == operation && descriptor.kernel.mode == mode
        })
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
        if let Some(existing) = self.launches.iter_mut().find(|existing| {
            existing.kernel.operation == descriptor.kernel.operation
                && existing.kernel.mode == descriptor.kernel.mode
        }) {
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

    pub fn has_operation(
        &self,
        layer: usize,
        operation: KernelOperation,
        mode: ExecutionMode,
    ) -> bool {
        self.layer(layer)
            .and_then(|plan| plan.operation(operation, mode))
            .is_some()
    }
}

// ── Provider manifest ─────────────────────────────────────────────────

/// Execution modes published for one semantic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationCapability {
    pub operation: KernelOperation,
    pub modes: ExecutionModeSet,
    pub deterministic_modes: ExecutionModeSet,
}

impl OperationCapability {
    pub const fn new(operation: KernelOperation, modes: ExecutionModeSet) -> Self {
        Self {
            operation,
            modes,
            deterministic_modes: ExecutionModeSet::NONE,
        }
    }

    pub const fn deterministic_in(mut self, modes: ExecutionModeSet) -> Self {
        self.deterministic_modes = modes;
        self
    }

    pub fn supports(self, requirement: OperationRequirement) -> bool {
        self.operation == requirement.operation
            && self.modes.contains(requirement.mode)
            && (!requirement.deterministic || self.deterministic_modes.contains(requirement.mode))
    }
}

/// Static metadata about a kernel provider, registered at startup.
#[derive(Debug, Clone)]
pub struct ProviderManifest {
    pub id: KernelProviderId,
    pub name: &'static str,
    pub operations: Vec<OperationCapability>,
}

impl ProviderManifest {
    pub fn new(
        id: KernelProviderId,
        name: &'static str,
        operations: impl Into<Vec<OperationCapability>>,
    ) -> Self {
        Self {
            id,
            name,
            operations: operations.into(),
        }
    }

    pub fn supports(&self, operation: KernelOperation, mode: ExecutionMode) -> bool {
        self.supports_requirement(OperationRequirement::new(operation, mode))
    }

    pub fn supports_requirement(&self, requirement: OperationRequirement) -> bool {
        self.operations
            .iter()
            .any(|capability| capability.supports(requirement))
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

    pub fn register(&mut self, manifest: ProviderManifest) {
        if let Some(existing) = self
            .providers
            .iter_mut()
            .find(|entry| entry.id == manifest.id)
        {
            *existing = manifest;
        } else {
            self.providers.push(manifest);
        }
    }

    pub fn is_available(&self, id: KernelProviderId) -> bool {
        self.providers.iter().any(|m| m.id == id)
    }

    pub fn supports(
        &self,
        id: KernelProviderId,
        operation: KernelOperation,
        mode: ExecutionMode,
    ) -> bool {
        self.supports_requirement(id, OperationRequirement::new(operation, mode))
    }

    pub fn supports_requirement(
        &self,
        id: KernelProviderId,
        requirement: OperationRequirement,
    ) -> bool {
        self.providers
            .iter()
            .find(|manifest| manifest.id == id)
            .is_some_and(|manifest| manifest.supports_requirement(requirement))
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
        let id = KernelId::new(
            KernelProviderId::CUDA_CUTLASS,
            KernelOperation::AttentionHcPre,
            ExecutionMode::Inference,
        );
        assert_eq!(id.provider, KernelProviderId::CUDA_CUTLASS);
        assert_eq!(id.operation, KernelOperation::AttentionHcPre);
        assert_eq!(id.mode, ExecutionMode::Inference);
        assert_eq!(id.phase(), KernelPhase::HcPre);
        assert_eq!(id.variant, 0);

        let id_v2 = id.with_variant(1);
        assert_eq!(id_v2.variant, 1);
        assert_eq!(id.variant, 0);
    }

    #[test]
    fn semantic_operations_use_expected_phases() {
        assert_eq!(
            KernelOperation::GroupedFp4Moe.phase(),
            KernelPhase::GroupedMoe
        );
        for operation in [
            KernelOperation::MainProjectNorm,
            KernelOperation::HybridMlaAttention,
            KernelOperation::ProposalHead,
        ] {
            assert_eq!(operation.phase(), KernelPhase::ProposalAttachment);
        }
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
    fn launch_descriptor_flags() {
        let id = KernelId::new(
            KernelProviderId::BUILTIN,
            KernelOperation::Embed,
            ExecutionMode::Inference,
        );
        let desc = LaunchDescriptor::new(id, (1, 1, 1), (256, 1, 1));
        assert!(!desc.is_capture_safe());

        let desc_safe = desc.capture_safe().with_shared_mem(4096);
        assert!(desc_safe.is_capture_safe());
        assert_eq!(desc_safe.shared_mem_bytes, 4096);
    }

    #[test]
    fn layer_kernel_plan_binds_operations_not_phases() {
        let mut plan = LayerKernelPlan::new();
        let attention_id = KernelId::new(
            KernelProviderId::BUILTIN,
            KernelOperation::AttentionHcPre,
            ExecutionMode::Inference,
        );
        let feed_forward_id = KernelId::new(
            KernelProviderId::BUILTIN,
            KernelOperation::FeedForwardHcPre,
            ExecutionMode::Inference,
        );
        plan.set_operation(LaunchDescriptor::new(attention_id, (1, 1, 1), (128, 1, 1)));
        plan.set_operation(LaunchDescriptor::new(
            feed_forward_id,
            (2, 1, 1),
            (128, 1, 1),
        ));

        assert_eq!(plan.operations_in_phase(KernelPhase::HcPre).count(), 2);
        assert_eq!(
            plan.operation(KernelOperation::AttentionHcPre, ExecutionMode::Inference,)
                .unwrap()
                .grid[0],
            1
        );

        // Replace only the exact operation, not another launch in the phase.
        plan.set_operation(LaunchDescriptor::new(attention_id, (3, 1, 1), (128, 1, 1)));
        assert_eq!(plan.operations_in_phase(KernelPhase::HcPre).count(), 2);
        assert_eq!(
            plan.operation(KernelOperation::AttentionHcPre, ExecutionMode::Inference,)
                .unwrap()
                .grid[0],
            3
        );
        assert_eq!(
            plan.operation(KernelOperation::FeedForwardHcPre, ExecutionMode::Inference,)
                .unwrap()
                .grid[0],
            2
        );
    }

    #[test]
    fn operation_modes_are_independent_bindings() {
        let mut plan = LayerKernelPlan::new();
        for (mode, grid_x) in [(ExecutionMode::Inference, 1), (ExecutionMode::Backward, 2)] {
            plan.set_operation(LaunchDescriptor::new(
                KernelId::new(KernelProviderId::BUILTIN, KernelOperation::Embed, mode),
                (grid_x, 1, 1),
                (128, 1, 1),
            ));
        }

        assert_eq!(
            plan.operation(KernelOperation::Embed, ExecutionMode::Inference)
                .unwrap()
                .grid[0],
            1
        );
        assert_eq!(
            plan.operation(KernelOperation::Embed, ExecutionMode::Backward)
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
            KernelId::new(
                KernelProviderId::BUILTIN,
                KernelOperation::Embed,
                ExecutionMode::Inference,
            ),
            (1, 1, 1),
            (128, 1, 1),
        );
        plan.layer_mut(0).unwrap().set_operation(launch);
        assert!(plan.has_operation(0, KernelOperation::Embed, ExecutionMode::Inference));
        assert!(!plan.has_operation(0, KernelOperation::Embed, ExecutionMode::Backward));
        assert!(!plan.has_operation(0, KernelOperation::Router, ExecutionMode::Inference));
    }

    #[test]
    fn provider_registry_checks_mode_and_determinism() {
        let manifest = ProviderManifest::new(
            KernelProviderId::new(7),
            "test-provider",
            vec![
                OperationCapability::new(
                    KernelOperation::Embed,
                    ExecutionModeSet::INFERENCE.union(ExecutionModeSet::TRAINING_FORWARD),
                )
                .deterministic_in(ExecutionModeSet::INFERENCE),
            ],
        );
        let mut registry = ProviderRegistry::new();
        registry.register(manifest);

        assert!(registry.is_available(KernelProviderId::new(7)));
        assert!(registry.supports(
            KernelProviderId::new(7),
            KernelOperation::Embed,
            ExecutionMode::Inference,
        ));
        assert!(registry.supports(
            KernelProviderId::new(7),
            KernelOperation::Embed,
            ExecutionMode::TrainingForward,
        ));
        assert!(!registry.supports(
            KernelProviderId::new(7),
            KernelOperation::Embed,
            ExecutionMode::Backward,
        ));
        assert!(
            registry.supports_requirement(
                KernelProviderId::new(7),
                OperationRequirement::new(KernelOperation::Embed, ExecutionMode::Inference)
                    .deterministic(),
            )
        );
        assert!(
            !registry.supports_requirement(
                KernelProviderId::new(7),
                OperationRequirement::new(KernelOperation::Embed, ExecutionMode::TrainingForward)
                    .deterministic(),
            )
        );
    }

    #[test]
    fn weight_layout_repr() {
        let layout = WeightLayout::RowMajor;
        assert_eq!(layout as u8, 0);
        let layout = WeightLayout::Fp4PackedExpertMajor;
        assert_eq!(layout as u8, 2);
    }
}
