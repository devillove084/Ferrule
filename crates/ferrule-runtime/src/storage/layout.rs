//! Storage layout — how bytes are interpreted.
//!
//! Layout is semantic enough for validation and backend selection, but not
//! model-family-specific. DSV4 "layer 17 expert 42" is model-binding metadata;
//! storage only sees an expert bundle object with size/layout/locators.

/// Semantic layout tag for a storage object's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageLayout {
    /// Raw bytes, no structure.
    Bytes,
    /// A tensor with dtype, shape, and stride.
    Tensor(TensorLayout),
    /// An expert bundle (gate + up + down).
    ExpertBundle(ExpertBundleLayout),
    /// A KV cache page.
    KvPage(KvPageLayout),
    /// Backend-specific opaque layout.
    Opaque { tag: String },
}

/// Tensor layout metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

/// Expert bundle layout — gate, up, down matrix shapes and quant format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBundleLayout {
    pub gate_shape: Vec<usize>,
    pub up_shape: Vec<usize>,
    pub down_shape: Vec<usize>,
    pub quant_format: String,
}

/// KV cache page layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPageLayout {
    pub head_dim: usize,
    pub num_heads: usize,
    pub page_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_layout_eq() {
        assert_eq!(StorageLayout::Bytes, StorageLayout::Bytes);
    }

    #[test]
    fn tensor_layout_construction() {
        let layout = TensorLayout {
            dtype: "BF16".into(),
            shape: vec![4096, 4096],
            stride: vec![4096, 1],
        };
        assert_eq!(layout.dtype, "BF16");
        assert_eq!(layout.shape.len(), 2);
    }

    #[test]
    fn expert_bundle_layout_construction() {
        let layout = ExpertBundleLayout {
            gate_shape: vec![2048, 7168],
            up_shape: vec![2048, 7168],
            down_shape: vec![7168, 2048],
            quant_format: "FP4_E2M1_E8M0".into(),
        };
        assert_eq!(layout.quant_format, "FP4_E2M1_E8M0");
    }

    #[test]
    fn kv_page_layout_construction() {
        let layout = KvPageLayout {
            head_dim: 128,
            num_heads: 32,
            page_size: 16,
        };
        assert_eq!(layout.page_size, 16);
    }

    #[test]
    fn opaque_layout_tagged() {
        let layout = StorageLayout::Opaque {
            tag: "cuda-fp4-artifact".into(),
        };
        if let StorageLayout::Opaque { tag } = &layout {
            assert_eq!(tag, "cuda-fp4-artifact");
        } else {
            panic!("expected Opaque");
        }
    }
}
