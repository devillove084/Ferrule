//! Family-neutral, inference-only transformer descriptions and state dictionaries.

pub mod attention;
mod checkpoint;
mod components;
pub mod connection;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
pub mod expert_parallel;
mod forward;
mod materialize;
mod operators;
pub mod parallel;
#[cfg(feature = "cuda")]
mod parallel_transfer;
pub mod proposal;
mod recipe;
mod segment;
mod standard;
mod state_dict;

pub use checkpoint::{
    BoundDecoderMaterializationSource, BoundDecoderResources, BoundDecoderSourceCatalogs,
    BoundExpertCatalog, BoundMatrix, DecoderLoadOptions, HFDecoderCheckpoint,
    parameter_resource_id,
};
pub use components::{
    Attention, DecoderAttachmentSpec, DecoderLayer, DecoderModelParts, DecoderModelSpec,
    DescriptorError, Embedding, FeedForward, GqaAttention, HyperConnectionHeadSpec, HyperResidual,
    Linear, MlaAttention, MlaAttentionLayout, MlaCompressorSpec, MlaDimensions, MlaIndexerSpec,
    MlaQueryProjection, Moe, MoeRouterSpec, ProposalAttachmentParts, ProposalAttachmentSpec,
    ProposalHeadsSpec, Residual, RmsNorm, RotaryEmbedding, RotaryPairing, RotaryRegion,
    RotaryScaling, RouterScoreFunction, RouterSelection, SharedKvMlaDimensions, SharedKvMlaLayout,
    SwiGlu,
};
pub use connection::{
    HyperConnection, HyperConnectionConfig, HyperConnectionHead, HyperConnectionHeadWeights,
    HyperConnectionPhase, HyperConnectionStage, HyperConnectionWeights, PreparedHyperConnection,
    PreparedHyperConnectionHead, PreparedHyperConnectionWeights,
};
pub use expert_parallel::{
    CpuReferenceExpertWorker, ExpertDispatchContext, ExpertDispatchLimits, ExpertDispatchPlan,
    ExpertPlacement, ExpertResult, ExpertToken, ExpertTokenBucket, ExpertWorker,
};
pub use forward::{
    AddResidual, Connected, HyperReduction, LayerMode, LayerRequest, MtpTap, NoPostLayerTap,
    NoReduction, OutputPipeline, OwnedArenaLease, Pending, Poll, PreparedTransformer, Step,
    TransformerContinuation, TransformerForwardExecutor, TransformerLayer, TransformerModule,
};
#[cfg(feature = "cuda")]
pub use materialize::PreparedCudaLinear;
pub use materialize::{
    LayerWeightCache, MemoryLayerWeightCache, PreparedDecoder, PreparedDecoderAttachment,
    PreparedDecoderGeneration, PreparedEmbedding, PreparedInteger, PreparedLinear, PreparedNorm,
    PreparedParameter, PreparedRope, StateDictMaterializer,
};
#[cfg(feature = "cuda")]
pub use operators::CudaRows;
pub use operators::{
    CpuStandardDecoderOperators, ExpertAvailability, ExpertProvider, ExpertSwiGluOperator,
    GqaMetadata, GqaRequest, HostRows, KvAppendRequest, KvHistory, KvView, OperatorProgress,
    OperatorWaiting, PreparedSwiGlu, RouterRoutes, Rows, RowsArenaId, RowsDType, RowsDevice,
    RowsShape, StandardDecoderKvView, StandardDecoderOperators, UnsupportedOperator,
};
pub use recipe::{DecoderRecipe, DecoderRecipeError, DecoderRecipeOutput, SyntheticDecoderRecipe};
pub use segment::{
    DeviceSegmentInput, DeviceSegmentOutput, LayerSegmentPlan, SegmentError, SegmentInput,
    SegmentOutput, SegmentResult, SegmentStage, StandardDecoderSegment,
};
pub use standard::{
    CpuGqaMoeModule, CpuTransformerHidden, PreparedCpuOutput, PreparedFeedForwardBlock,
    PreparedGqaBlock, PreparedGqaMoeLayer, PreparedStandardOutput, StandardTransformerHidden,
};
pub use state_dict::{
    BindingIssue, BoundParameter, BoundStateDict, BoundTensorPart, ExactNameMapper,
    ExternalTensorMeta, NameMapError, NameMapper, NameMapping, StateDictBindError, StateDictBinder,
    StateDictSchema, StateDictSchemaBuilder, StateDictSchemaError, TensorTransform,
};

#[cfg(feature = "cuda")]
pub use standard::cuda::{
    CudaExpertParallelRoutedExecutor, CudaExpertWorker, CudaHostRoutedExecutor,
    CudaStandardDecoderOperators, CudaStandardDecoderSegment, CudaStandardKvBinding,
};
