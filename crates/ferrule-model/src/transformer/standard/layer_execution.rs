//! Private owner dispatch: segment loops keep their ordinary trait-object path,
//! while CUDA owners opt into a lexical root scope around the same layer math.
use super::*;

pub(in crate::transformer) enum LayerOperators<'a> {
    Standard(&'a mut dyn StandardDecoderOperators),
    #[cfg(feature = "cuda")]
    Cuda(&'a mut cuda::CudaStandardDecoderOperators),
}

impl LayerOperators<'_> {
    pub(in crate::transformer) fn operators(&mut self) -> &mut dyn StandardDecoderOperators {
        match self {
            Self::Standard(operators) => *operators,
            #[cfg(feature = "cuda")]
            Self::Cuda(operators) => *operators,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::transformer) fn execute_layer(
        &mut self,
        materializer: &StateDictMaterializer,
        layer_index: usize,
        attention: hybrid::AttentionBlockRef<'_>,
        feed_forward: &PreparedFeedForwardBlock,
        kv_layer: usize,
        hidden: &mut CpuTransformerHidden,
        kv: &mut dyn KvView,
        states: &mut [&mut dyn crate::decoder::StandardSequenceState],
        routed_execution: Option<&mut RoutedLayerExecution<'_>>,
    ) -> Result<()> {
        match self {
            Self::Standard(operators) => execute_standard_layer_inner(
                *operators,
                materializer,
                layer_index,
                attention,
                feed_forward,
                kv_layer,
                hidden,
                kv,
                states,
                routed_execution,
            ),
            #[cfg(feature = "cuda")]
            Self::Cuda(operators) => operators.execute_root_layer(
                materializer,
                layer_index,
                attention,
                feed_forward,
                kv_layer,
                hidden,
                kv,
                states,
                routed_execution,
            ),
        }
    }
}
