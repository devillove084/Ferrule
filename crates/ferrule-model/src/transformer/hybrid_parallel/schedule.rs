//! Versioned model-owned collective schedule. Runtime translates these sites;
//! it must not reconstruct the decoder loop or the agreement packet length.
use super::*;
use crate::transformer::LayerSegmentPlan;

/// V2 removes owner-local execution generation and adds an explicit version word.
/// Integers are four little-endian u16 limbs represented exactly as F32.
pub const HYBRID_AGREEMENT_VERSION: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HybridCollectiveKind {
    AllGather = 0,
    ReduceSum = 1,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HybridCollectiveSemantic {
    Agreement = 0,
    AttentionOutput = 1,
    DenseMlpDown = 2,
    SharedExpertDown = 3,
    RoutedExpertOutput = 4,
    OutputHead = 5,
    Completion = 6,
}
/// Number of logical elements sent by EACH rank. Gather width includes padding.
/// Legacy V2 schedules carry F32; the explicit Mixed block contract carries BF16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridCollectiveShape {
    pub fixed: usize,
    pub per_sequence: usize,
    pub per_row: usize,
}
impl HybridCollectiveShape {
    pub fn elements(&self, sequences: usize, rows: usize) -> Result<usize> {
        self.per_sequence
            .checked_mul(sequences)
            .and_then(|n| n.checked_add(self.fixed))
            .and_then(|n| {
                self.per_row
                    .checked_mul(rows)
                    .and_then(|r| n.checked_add(r))
            })
            .filter(|n| {
                n.checked_mul(4)
                    .is_some_and(|bytes| bytes <= isize::MAX as usize)
            })
            .ok_or_else(|| invalid("collective shape exceeds address space"))
    }
    fn rows(width: usize) -> Self {
        Self {
            fixed: 0,
            per_sequence: 0,
            per_row: width,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridCollectiveSite {
    pub ordinal: u64,
    pub global_layer: Option<usize>,
    pub semantic: HybridCollectiveSemantic,
    pub kind: HybridCollectiveKind,
    pub shape: HybridCollectiveShape,
}
// This recipe is the sole source of both packet serialization and its shape.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AgreementEncoding {
    prefix: Vec<u64>,
    sequence_fields: [SequenceField; 4],
    row_fields: [RowField; 3],
    hidden_width: Option<usize>,
    next_layer: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceField {
    Session,
    Context,
    End,
    Phase,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowField {
    Token,
    Position,
    Sequence,
}
impl AgreementEncoding {
    fn shape(&self) -> Result<HybridCollectiveShape> {
        let fixed = self
            .prefix
            .len()
            .checked_add(2)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("agreement prefix overflow"))?;
        let per_sequence = self
            .sequence_fields
            .len()
            .checked_mul(4)
            .ok_or_else(|| invalid("agreement sequence overflow"))?;
        let per_row = self
            .row_fields
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_add(self.hidden_width.unwrap_or(0)))
            .ok_or_else(|| invalid("agreement row overflow"))?;
        Ok(HybridCollectiveShape {
            fixed,
            per_sequence,
            per_row,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridCollectiveSchedule {
    sites: Vec<HybridCollectiveSite>,
    agreement: AgreementEncoding,
    degree: usize,
}
impl HybridCollectiveSchedule {
    pub fn agreement_version(&self) -> u64 {
        HYBRID_AGREEMENT_VERSION
    }
    /// Logical sites consumed by all degrees, including local TP1 execution.
    pub fn sites(&self) -> &[HybridCollectiveSite] {
        &self.sites
    }
    /// TP1 performs no wire calls, including agreement and completion.
    pub fn wire_sites(&self) -> &[HybridCollectiveSite] {
        if self.degree == 1 { &[] } else { &self.sites }
    }
    pub fn degree(&self) -> usize {
        self.degree
    }
    /// Stable metadata words for registration/comparison, not authentication or
    /// model identity. Header: `[version, degree, logical_site_count]`; each site:
    /// `[ordinal, global_layer_or_u64_MAX, semantic, kind, fixed, per_sequence, per_row]`.
    /// For actual transport admission use `wire_sites()`, which is empty for TP1.
    pub fn encode(&self) -> Vec<u64> {
        let mut out = vec![
            HYBRID_AGREEMENT_VERSION,
            self.degree as u64,
            self.sites.len() as u64,
        ];
        for site in &self.sites {
            out.extend([
                site.ordinal,
                site.global_layer.map_or(u64::MAX, |x| x as u64),
                site.semantic as u64,
                site.kind as u64,
                site.shape.fixed as u64,
                site.shape.per_sequence as u64,
                site.shape.per_row as u64,
            ]);
        }
        out
    }
    /// Encode logical input agreement using the same recipe as the first site's
    /// shape. This excludes owner-local generations; callers must independently
    /// validate the batch against their entered KV and local sequence states.
    pub fn encode_agreement(
        &self,
        batch: &crate::decoder::PackedDecoderBatch,
        sessions: &[u64],
        input: &crate::transformer::SegmentInput,
    ) -> Result<Vec<f32>> {
        use crate::transformer::{RowsDType, SegmentInput};
        if sessions.len() != batch.sequences().len() {
            return Err(invalid("agreement session count mismatch"));
        }
        let hidden = match (self.agreement.hidden_width, input) {
            (None, SegmentInput::Tokens) => None,
            (Some(width), SegmentInput::Hidden { next_layer, rows })
                if *next_layer == self.agreement.next_layer
                    && rows.shape().rows() == batch.len()
                    && rows.shape().width() == width
                    && rows.dtype() == RowsDType::F32 =>
            {
                Some(rows.values())
            }
            _ => {
                return Err(invalid(
                    "agreement hidden boundary/shape/dtype or embedding ownership mismatch",
                ));
            }
        };
        let count = self
            .agreement
            .shape()?
            .elements(sessions.len(), batch.len())?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| invalid("agreement allocation failed"))?;
        let push = |values: &mut Vec<f32>, n: u64| {
            for shift in [0, 16, 32, 48] {
                values.push(((n >> shift) & 0xffff) as f32);
            }
        };
        for &n in &self.agreement.prefix {
            push(&mut values, n);
        }
        push(&mut values, batch.len() as u64);
        push(&mut values, sessions.len() as u64);
        for (&id, sequence) in sessions.iter().zip(batch.sequences()) {
            for field in self.agreement.sequence_fields {
                push(
                    &mut values,
                    match field {
                        SequenceField::Session => id,
                        SequenceField::Context => sequence.context_len() as u64,
                        SequenceField::End => sequence.sequence_len() as u64,
                        SequenceField::Phase => match sequence.phase() {
                            ferrule_common::execution::ForwardPhase::Prefill => 0,
                            ferrule_common::execution::ForwardPhase::Decode => 1,
                        },
                    },
                );
            }
        }
        for row in 0..batch.len() {
            for field in self.agreement.row_fields {
                push(
                    &mut values,
                    match field {
                        RowField::Token => batch.token_ids()[row] as u64,
                        RowField::Position => batch.positions()[row] as u64,
                        RowField::Sequence => batch.row_to_sequence()[row] as u64,
                    },
                );
            }
        }
        if let Some(hidden) = hidden {
            values.extend_from_slice(hidden);
        }
        if values.len() != count || values.iter().any(|x| !x.is_finite()) {
            return Err(invalid("invalid agreement packet"));
        }
        Ok(values)
    }
}
impl HybridParallelPlan {
    pub fn collective_schedule(
        &self,
        segment: &LayerSegmentPlan,
    ) -> Result<HybridCollectiveSchedule> {
        self.validate_rank_execution()?;
        if segment.layers() != self.schema.layer_range()
            || segment.total_layers() != self.spec.layers().len()
        {
            return Err(invalid("collective schedule segment mismatch"));
        }
        let scope = self.scope();
        let mut prefix = vec![
            HYBRID_AGREEMENT_VERSION,
            scope.topology.get() as u64,
            scope.replica as u64,
            scope.stage as u64,
            segment.layers().start as u64,
            segment.layers().end as u64,
            self.tensor_members.len() as u64,
        ];
        prefix.extend(self.tensor_members.iter().map(|m| m.get() as u64));
        prefix.push(u64::from(self.expert_attachment().is_some()));
        if let Some(ep) = self.expert_attachment() {
            prefix.extend([ep.source_rank.get() as u64, ep.members.len() as u64]);
            prefix.extend(ep.members.iter().map(|m| m.get() as u64));
            for layer in &self.layers {
                prefix.extend([layer.routed_experts as u64, layer.top_k as u64]);
                for expert in 0..layer.routed_experts {
                    prefix.push(
                        ep.placement
                            .owner(layer.global_layer, expert)
                            .ok_or_else(|| invalid("missing admitted expert"))?
                            .get() as u64,
                    );
                }
            }
        }
        prefix.push(u64::from(!segment.owns_embedding()));
        if !segment.owns_embedding() {
            prefix.push(segment.layers().start as u64);
        }
        let agreement = AgreementEncoding {
            prefix,
            sequence_fields: [
                SequenceField::Session,
                SequenceField::Context,
                SequenceField::End,
                SequenceField::Phase,
            ],
            row_fields: [RowField::Token, RowField::Position, RowField::Sequence],
            hidden_width: (!segment.owns_embedding()).then_some(self.width),
            next_layer: segment.layers().start,
        };
        let mut sites = Vec::new();
        let mut push = |layer, semantic, kind, shape| {
            sites.push(HybridCollectiveSite {
                ordinal: sites.len() as u64,
                global_layer: layer,
                semantic,
                kind,
                shape,
            });
        };
        use HybridCollectiveKind::{AllGather, ReduceSum};
        use HybridCollectiveSemantic::*;
        push(None, Agreement, AllGather, agreement.shape()?);
        for layer in &self.layers {
            push(
                Some(layer.global_layer),
                AttentionOutput,
                ReduceSum,
                HybridCollectiveShape::rows(self.width),
            );
            if layer.dense_mlp.is_some() {
                push(
                    Some(layer.global_layer),
                    DenseMlpDown,
                    ReduceSum,
                    HybridCollectiveShape::rows(self.width),
                );
            }
            if layer.shared_expert.is_some() {
                push(
                    Some(layer.global_layer),
                    SharedExpertDown,
                    ReduceSum,
                    HybridCollectiveShape::rows(self.width),
                );
            }
            // The shared scalar gate is local math after SharedExpertDown; no wire site.
            if layer.routed_experts > 0 {
                push(
                    Some(layer.global_layer),
                    RoutedExpertOutput,
                    ReduceSum,
                    HybridCollectiveShape::rows(self.width),
                );
            }
        }
        if segment.owns_output() {
            push(
                None,
                OutputHead,
                AllGather,
                HybridCollectiveShape::rows(
                    self.spec.vocab_size().div_ceil(self.tensor_members.len()),
                ),
            );
        }
        push(
            None,
            Completion,
            ReduceSum,
            HybridCollectiveShape {
                fixed: 1,
                per_sequence: 0,
                per_row: 0,
            },
        );
        Ok(HybridCollectiveSchedule {
            sites,
            agreement,
            degree: self.tensor_members.len(),
        })
    }
}
