//! Subordinate read record in the existing transaction Entry, not a page owner.
//! Driver-free mapping/fence protocol; CUDA supplies opaque completion evidence.
use super::plan::{Plan, error};
use crate::cpu::PagedKvBatch;
use ferrule_common::Result;
use ferrule_common::execution::KvPageId;
use ferrule_common::host_kv::HostKvAppendReadWitness;
use std::cell::Cell;
use std::rc::Rc;

/// Contract for the trusted DMA adapter: disjoint exact ranges, identical Q
/// geometry/position/mask, and unnormalized partials. Origin of the numeric
/// operands must still be established by the transaction's consumer fences.
pub(super) fn merge_append_partials(
    witness: &HostKvAppendReadWitness,
    config: crate::cpu::StreamingF32GqaConfig,
    prefix_range: std::ops::Range<usize>,
    prefix: &crate::cpu::StreamingF32GqaPartial,
    tail_range: std::ops::Range<usize>,
    tail: &crate::cpu::StreamingF32GqaPartial,
) -> Result<crate::cpu::StreamingF32GqaPartial> {
    let position = witness.source().committed_tokens;
    witness
        .validate_partial_partition(position, prefix_range, tail_range)
        .map_err(|e| error(e.to_string()))?;
    config.validate()?;
    if config.rows != 1
        || witness
            .source()
            .schema
            .planes()
            .iter()
            .any(|p| Some(p.elements_per_token) != config.kv_elements(1).ok())
    {
        return Err(error("append partial geometry differs from prefix schema"));
    }
    let mut total = crate::cpu::StreamingF32GqaPartial::empty(config, position as u64)?;
    total.merge(prefix)?;
    total.merge(tail)?;
    Ok(total)
}

#[derive(Debug, PartialEq, Eq)]
struct Mapping {
    source: KvPageId,
    original: u32,
    original_epoch: u64,
    projected: KvPageId,
    slot: u32,
    epoch: u64,
}

/// Logical source and layer bound to one physical transaction incarnation.
/// This request is checked against custody; it is not itself a capability.
#[derive(Clone, Copy)]
pub(super) struct PrefixReadBinding<'a> {
    pub witness: &'a HostKvAppendReadWitness,
    pub layer: usize,
    pub nonce: u64,
    pub view_epoch: u64,
}

/// Cannot be constructed/cloned by consumers. Drop does not remove the record.
#[derive(Debug)]
pub(super) struct PrefixReadHandle {
    owner: Rc<()>,
    serial: u64,
    nonce: u64,
    view_epoch: u64,
}

enum Fence<F> {
    NoSubmit,
    Pending,
    Recorded(F),
}
struct Record<F> {
    serial: u64,
    nonce: u64,
    view_epoch: u64,
    witness: HostKvAppendReadWitness,
    layer: usize,
    mappings: Vec<Mapping>,
    writes: Vec<(KvPageId, u32, u64)>,
    producer: F,
    dma: Fence<F>,
    consumer: Fence<F>,
    unknown: Cell<bool>,
}

/// At most one read per Entry. Real DMA/consumer events stay in that Entry even
/// if the external capability is dropped or the append is being aborted.
pub(super) struct PrefixReadCustody<F> {
    identity: Option<Rc<()>>,
    binding: Option<HostKvAppendReadWitness>,
    next: u64,
    record: Option<Record<F>>,
}
impl<F> Default for PrefixReadCustody<F> {
    fn default() -> Self {
        Self {
            identity: None,
            binding: None,
            next: 1,
            record: None,
        }
    }
}
impl<F> PrefixReadCustody<F> {
    pub fn is_unknown(&self) -> bool {
        self.record.as_ref().is_some_and(|r| r.unknown.get())
    }
    pub fn ensure_idle(&self) -> Result<()> {
        if self.record.is_some() {
            return Err(error("transaction prefix read still owns custody"));
        }
        Ok(())
    }
    pub fn derive(
        &mut self,
        binding: PrefixReadBinding<'_>,
        batch: &PagedKvBatch,
        plan: &Plan,
        epoch: impl Fn(u32) -> Option<u64>,
        producer: F,
    ) -> Result<PrefixReadHandle> {
        let PrefixReadBinding {
            witness,
            layer,
            nonce,
            view_epoch,
        } = binding;
        self.ensure_idle()?;
        if self.binding.as_ref().is_some_and(|bound| bound != witness) {
            return Err(error("transaction prefix logical binding changed"));
        }
        let next = self
            .next
            .checked_add(1)
            .ok_or_else(|| error("prefix read serial exhausted"))?;
        let source = witness.source();
        let page_size = source.schema.page_size();
        if nonce == 0
            || view_epoch == 0
            || source
                .schema
                .planes()
                .iter()
                .any(|p| layer >= p.layer_count)
            || batch.sequences.len() != 1
            || batch.row_sequence_ids.as_ref() != [0]
            || batch.row_positions.as_ref() != [source.committed_tokens]
            || batch.sequences[0].sequence_len != witness.reservation().positions.end
            || batch.sequences[0].block_table.as_ref() != witness.pages()
        {
            return Err(error(
                "prefix read requires exact single-query transaction batch/layer",
            ));
        }
        plan.packed_slots(batch, page_size)?;
        let writes = plan
            .staged
            .iter()
            .map(|(&p, &s)| {
                Ok((
                    p,
                    s,
                    epoch(s).ok_or_else(|| error("missing write slot epoch"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut mappings = Vec::with_capacity(source.pages.len());
        for (i, &page) in source.pages.iter().enumerate() {
            let original = plan
                .original
                .get(&page)
                .copied()
                .flatten()
                .ok_or_else(|| error("prefix source is not protected original storage"))?;
            let projected = witness.pages()[i];
            let slot = plan
                .slot(projected)
                .ok_or_else(|| error("prefix projection absent"))?;
            // Even the mutable tail is read from original storage, not its
            // shadow. COW destination must have copied exactly this source.
            if (slot != original && plan.copies.get(&projected) != Some(&Some(original)))
                || (slot == original && plan.copies.contains_key(&projected))
                || writes.iter().any(|(_, s, _)| *s == original)
                || mappings.iter().any(|m: &Mapping| m.original == original)
            {
                return Err(error(
                    "prefix source aliases writes or wrong COW/shadow copy",
                ));
            }
            mappings.push(Mapping {
                source: page,
                original,
                original_epoch: epoch(original).ok_or_else(|| error("missing source epoch"))?,
                projected,
                slot,
                epoch: epoch(slot).ok_or_else(|| error("missing projected epoch"))?,
            });
        }
        let serial = self.next;
        self.next = next;
        let identity = self.identity.get_or_insert_with(|| Rc::new(())).clone();
        self.binding = Some(witness.clone());
        self.record = Some(Record {
            serial,
            nonce,
            view_epoch,
            witness: witness.clone(),
            layer,
            mappings,
            writes,
            producer,
            dma: Fence::NoSubmit,
            consumer: Fence::NoSubmit,
            unknown: Cell::new(false),
        });
        Ok(PrefixReadHandle {
            owner: identity,
            serial,
            nonce,
            view_epoch,
        })
    }
    fn record(&self, handle: &PrefixReadHandle) -> Result<&Record<F>> {
        let r = self
            .record
            .as_ref()
            .ok_or_else(|| error("prefix read absent"))?;
        if !self
            .identity
            .as_ref()
            .is_some_and(|id| Rc::ptr_eq(&handle.owner, id))
            || handle.serial != r.serial
            || handle.nonce != r.nonce
            || handle.view_epoch != r.view_epoch
        {
            return Err(error("foreign or stale prefix read capability"));
        }
        Ok(r)
    }
    pub fn validate(
        &self,
        handle: &PrefixReadHandle,
        binding: PrefixReadBinding<'_>,
        plan: &Plan,
        epoch: impl Fn(u32) -> Option<u64>,
    ) -> Result<()> {
        let PrefixReadBinding {
            witness,
            layer,
            nonce,
            view_epoch,
        } = binding;
        let r = self.record(handle)?;
        if r.unknown.get() {
            return Err(error("prefix read completion unknown"));
        }
        if &r.witness != witness
            || r.layer != layer
            || r.nonce != nonce
            || r.view_epoch != view_epoch
        {
            return Err(error(
                "prefix read source/layer/transaction incarnation changed",
            ));
        }
        if r.writes.len() != plan.staged.len()
            || r.writes
                .iter()
                .any(|&(p, s, e)| plan.staged.get(&p) != Some(&s) || epoch(s) != Some(e))
            || r.mappings.iter().any(|m| {
                plan.original.get(&m.source) != Some(&Some(m.original))
                    || plan.slot(m.projected) != Some(m.slot)
                    || epoch(m.original) != Some(m.original_epoch)
                    || epoch(m.slot) != Some(m.epoch)
                    || (m.slot != m.original
                        && plan.copies.get(&m.projected) != Some(&Some(m.original)))
            })
        {
            return Err(error("prefix mapping or physical slot epoch changed"));
        }
        Ok(())
    }
    /// The internal DMA adapter must call this BEFORE either submission.
    /// A failed/unrecorded submission cannot turn into a no-submit cancellation.
    pub fn begin_submission(&mut self, handle: &PrefixReadHandle) -> Result<()> {
        let r = self.record(handle)?;
        if r.unknown.get()
            || !matches!(&r.dma, Fence::NoSubmit)
            || !matches!(&r.consumer, Fence::NoSubmit)
        {
            return Err(error("prefix submission is not retryable"));
        }
        let r = self.record.as_mut().unwrap();
        r.dma = Fence::Pending;
        r.consumer = Fence::Pending;
        Ok(())
    }
    pub fn record_fences(&mut self, handle: &PrefixReadHandle, dma: F, consumer: F) -> Result<()> {
        let r = self.record(handle)?;
        if r.unknown.get()
            || !matches!(&r.dma, Fence::Pending)
            || !matches!(&r.consumer, Fence::Pending)
        {
            return Err(error("prefix fences cannot replace existing evidence"));
        }
        let r = self.record.as_mut().unwrap();
        r.dma = Fence::Recorded(dma);
        r.consumer = Fence::Recorded(consumer);
        Ok(())
    }
    pub fn mark_unknown(&self, handle: &PrefixReadHandle) -> Result<()> {
        self.record(handle)?.unknown.set(true);
        Ok(())
    }
    /// Used by both normal finish and abort cleanup. No fresh event or boolean
    /// ACK is accepted. An event-query error is sticky for this registration.
    pub fn finish(
        &mut self,
        handle: &PrefixReadHandle,
        mut query: impl FnMut(&F) -> Result<bool>,
    ) -> Result<bool> {
        let unknown = self.record(handle)?.unknown.get();
        if unknown {
            return Err(error("prefix read completion unknown"));
        }
        let ready = (|| {
            let r = self.record(handle)?;
            let producer = query(&r.producer)?;
            let mut poll = |f: &Fence<F>| match f {
                Fence::NoSubmit => Ok(true),
                Fence::Pending => Err(error("prefix submission fence is unknown")),
                Fence::Recorded(f) => query(f),
            };
            let dma = poll(&r.dma)?;
            let consumer = poll(&r.consumer)?;
            Ok(producer && dma && consumer)
        })();
        match ready {
            Ok(true) => {
                self.record = None;
                Ok(true)
            }
            Ok(false) => Ok(false),
            Err(e) => {
                self.record
                    .as_ref()
                    .expect("validated record")
                    .unknown
                    .set(true);
                Err(e)
            }
        }
    }
}
