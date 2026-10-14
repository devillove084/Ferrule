//! Backend-only physical placement contract, independent of CUDA and logical KV ownership.
//!
//! This is a subordinate custody record for an existing transaction Entry, not a
//! manager/registry. The default fixed-plane pool does not use independent pages.
//! A placement or slot reuse is never evidence of allocator or driver reclamation.

use ferrule_common::execution::KvPageId;
use ferrule_common::{Error, Result};
use std::ops::Range;
use std::rc::Rc;

fn error(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

#[derive(Debug, Clone)]
struct Identity(Rc<()>);
impl Identity {
    fn new() -> Self {
        Self(Rc::new(()))
    }
}
impl PartialEq for Identity {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Identity {}

/// Owner-local address space. No global ID counter or registry is required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageAllocationDomain(Identity);
impl Default for PageAllocationDomain {
    fn default() -> Self {
        Self::new()
    }
}
impl PageAllocationDomain {
    pub fn new() -> Self {
        Self(Identity::new())
    }
}

/// Allocation incarnation, binding incarnation and placement generation. A new
/// binding of the same physical slot must not resurrect its previous descriptors.
/// Not a device pointer, slot index, logical page ID or completion proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageAllocationIdentity {
    allocation: Identity,
    binding: Identity,
    generation: u64,
}
impl PageAllocationIdentity {
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageAllocationKind {
    /// One page is a range of a retained plane allocation; reclaim is slot-only.
    FixedPlane,
    /// One independently owned allocator grant, not necessarily a driver segment.
    SinglePage,
}

struct Allocation<B> {
    identity: Identity,
    domain: PageAllocationDomain,
    base: u64,
    requested_bytes: usize,
    granted_bytes: usize,
    kind: PageAllocationKind,
    _backing: B,
}

/// Retains the *whole* physical grant, including when placements name subranges.
/// Clones share the grant; dropping a placement does not imply a last-owner drop.
pub struct PageAllocation<B>(Rc<Allocation<B>>);
impl<B> Clone for PageAllocation<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<B> PageAllocation<B> {
    /// Adopt an actual allocator grant. No allocation is performed here.
    ///
    /// # Safety
    /// `backing` must retain the entire live allocation at `base` in `domain`
    /// until dropped. Sizes must describe that allocation (not a borrowed view).
    /// All grants composed together must use the same domain/address space.
    /// Create one identity per live grant and clone it for views; never re-adopt a view as a
    /// new grant. The caller must retain any outstanding pre-existing consumers.
    pub unsafe fn from_backing(
        backing: B,
        domain: PageAllocationDomain,
        base: u64,
        requested_bytes: usize,
        granted_bytes: usize,
        kind: PageAllocationKind,
    ) -> Result<Self> {
        let bytes = u64::try_from(granted_bytes).map_err(|_| error("grant size overflow"))?;
        if base == 0
            || requested_bytes == 0
            || granted_bytes < requested_bytes
            || base.checked_add(bytes).is_none()
        {
            return Err(error("invalid physical allocation extent"));
        }
        Ok(Self(Rc::new(Allocation {
            identity: Identity::new(),
            domain,
            base,
            requested_bytes,
            granted_bytes,
            kind,
            _backing: backing,
        })))
    }

    /// Bind a whole logical page/plane extent, including all of its layers.
    /// Single-page allocations cannot be disguised as fixed-plane suballocations.
    pub fn placement(
        &self,
        page: KvPageId,
        plane: usize,
        byte_range: Range<usize>,
        element_bytes: usize,
    ) -> Result<PagePhysicalPlacement<B>> {
        if byte_range.start >= byte_range.end
            || byte_range.end > self.0.requested_bytes
            || !element_bytes.is_power_of_two()
            || !byte_range.start.is_multiple_of(element_bytes)
            || !byte_range.end.is_multiple_of(element_bytes)
            || !self.0.base.is_multiple_of(element_bytes as u64)
            || (self.0.kind == PageAllocationKind::SinglePage
                && byte_range != (0..self.0.requested_bytes))
        {
            return Err(error("invalid page extent or element alignment"));
        }
        Ok(PagePhysicalPlacement {
            allocation: self.clone(),
            chunk: PagePhysicalChunk {
                identity: PageAllocationIdentity {
                    allocation: self.0.identity.clone(),
                    binding: Identity::new(),
                    generation: 1,
                },
                page,
                plane,
                byte_range,
            },
            element_bytes,
        })
    }
}

/// Metadata only: no pointer, allocation owner or authority to acknowledge work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagePhysicalChunk {
    identity: PageAllocationIdentity,
    page: KvPageId,
    plane: usize,
    byte_range: Range<usize>,
}
impl PagePhysicalChunk {
    pub fn identity(&self) -> &PageAllocationIdentity {
        &self.identity
    }
    pub fn page(&self) -> KvPageId {
        self.page
    }
    pub fn plane(&self) -> usize {
        self.plane
    }
    pub fn byte_range(&self) -> Range<usize> {
        self.byte_range.clone()
    }
}

pub struct PagePhysicalPlacement<B> {
    allocation: PageAllocation<B>,
    chunk: PagePhysicalChunk,
    element_bytes: usize,
}
impl<B> PagePhysicalPlacement<B> {
    pub fn descriptor(&self) -> &PagePhysicalChunk {
        &self.chunk
    }

    /// Byte offsets are relative to this page, not the plane or device address.
    /// Crossing a page boundary, empty chunks and partial elements are rejected.
    pub fn chunk(&self, offset: usize, bytes: usize) -> Result<PagePhysicalChunk> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| error("chunk overflow"))?;
        if bytes == 0
            || end > self.chunk.byte_range.len()
            || !offset.is_multiple_of(self.element_bytes)
            || !bytes.is_multiple_of(self.element_bytes)
        {
            return Err(error("chunk crosses page or element boundary"));
        }
        let mut chunk = self.chunk.clone();
        chunk.byte_range = self.chunk.byte_range.start + offset..self.chunk.byte_range.start + end;
        Ok(chunk)
    }

    fn overlaps(&self, other: &Self) -> bool {
        let absolute = |p: &Self| {
            p.allocation.0.base + p.chunk.byte_range.start as u64
                ..p.allocation.0.base + p.chunk.byte_range.end as u64
        };
        let a = absolute(self);
        let b = absolute(other);
        a.start < b.end && b.start < a.end
    }
}

/// Actual backend completion evidence, not an ACK tuple supplied by a client.
///
/// # Safety
/// `is_complete` may return true only after the fence's physical work is known
/// complete. Errors/unknown must never be converted to true. Implementations
/// must retain whatever is needed to query that evidence safely.
///
/// A safe implementation cannot invent an always-complete fence:
/// ```compile_fail,E0200
/// use ferrule_backend::physical_placement::PageCompletionFence;
/// struct Ack;
/// impl PageCompletionFence for Ack {
///     fn is_complete(&self) -> ferrule_common::Result<bool> { Ok(true) }
/// }
/// ```
pub unsafe trait PageCompletionFence {
    fn is_complete(&self) -> Result<bool>;
}

#[derive(Debug, Clone)]
pub struct PagePlacementSubmission {
    owner: Identity,
    submission: Identity,
    generation: u64,
}

/// Exact execution domain covered by a trusted completion token. A role tag
/// alone is not evidence; only the unsafe backend adapter can mint its token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageCompletionDomain {
    Producer,
    Dma,
    Consumer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionScope {
    Submission,
    FullDrain,
}

/// Opaque, non-cloneable backend evidence bound to one submission AND one domain.
/// Neither raw fences nor public number tuples are accepted by the custody API.
/// A compute-stream event must not be labeled DMA evidence unless the adapter
/// establishes its dependency on all exact transfer completions.
///
/// Safe code cannot mint evidence from a fence or relabel submission as drain:
/// ```compile_fail,E0133
/// use ferrule_backend::physical_placement::*;
/// fn forge<F: PageCompletionFence>(s: &PagePlacementSubmission, fence: F) {
///     PageCompletionToken::after_submission(s, PageCompletionDomain::Dma, fence);
/// }
/// ```
/// ```compile_fail,E0133
/// use ferrule_backend::physical_placement::*;
/// fn wash_unknown<F: PageCompletionFence>(s: &PagePlacementSubmission, fence: F) {
///     PageCompletionToken::after_full_drain(s, PageCompletionDomain::Dma, fence);
/// }
/// ```
/// ```compile_fail
/// use ferrule_backend::physical_placement::*;
/// fn replay<F: PageCompletionFence>(token: &PageCompletionToken<F>) -> PageCompletionToken<F> {
///     token.clone()
/// }
/// ```
pub struct PageCompletionToken<F: PageCompletionFence> {
    submission: PagePlacementSubmission,
    domain: PageCompletionDomain,
    scope: CompletionScope,
    fence: F,
}
impl<F: PageCompletionFence> PageCompletionToken<F> {
    /// # Safety
    /// The adapter must authenticate this exact submission's allocations and
    /// bind ALL work in `domain` to `fence`, including the last output reader.
    /// Completion on a different stream/domain, client ACK, enqueue success or
    /// metadata equality is insufficient. No work may be launched after this
    /// domain's final fence without a new submission.
    pub unsafe fn after_submission(
        submission: &PagePlacementSubmission,
        domain: PageCompletionDomain,
        fence: F,
    ) -> Self {
        Self {
            submission: submission.clone(),
            domain,
            scope: CompletionScope::Submission,
            fence,
        }
    }

    /// # Safety
    /// The adapter must authenticate this exact submission and establish a full
    /// drain of ALL work in `domain`, including failed launches with no returned
    /// ticket. An ordinary late completion token cannot be upgraded to a drain.
    /// This physical evidence does not clear logical Entry sticky Unknown.
    pub unsafe fn after_full_drain(
        submission: &PagePlacementSubmission,
        domain: PageCompletionDomain,
        fence: F,
    ) -> Self {
        Self {
            submission: submission.clone(),
            domain,
            scope: CompletionScope::FullDrain,
            fence,
        }
    }
}

/// Non-cloneable proof emitted only after producer, DMA and consumer completion.
/// It cannot be constructed from public metadata or replayed after slot reuse.
///
/// ```compile_fail
/// use ferrule_backend::physical_placement::PagePlacementProof;
/// fn replay(proof: &PagePlacementProof) -> PagePlacementProof { proof.clone() }
/// ```
/// ```compile_fail,E0423
/// use ferrule_backend::physical_placement::{PagePlacementProof, PagePlacementSubmission};
/// fn forge(submission: PagePlacementSubmission) -> PagePlacementProof {
///     PagePlacementProof(submission)
/// }
/// ```
#[derive(Debug)]
pub struct PagePlacementProof(PagePlacementSubmission);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Submitting,
    Pending,
    Unknown,
    /// Full physical drain allows release only; never resume an Unknown binding.
    Drained,
    Ready,
    Released,
}

/// Record-local retained grants, deduplicated across page/chunk views. Not a
/// global allocator snapshot: other owners can continue to hold these grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagePlacementMetrics {
    pub slot_reuses: u64,
    pub live_allocation_bytes: usize,
    /// Unavailable here. Only actual allocator/driver telemetry can fill this;
    /// even a last-owner drop can leave bytes cached or pending retirement.
    pub driver_freed_bytes: Option<u64>,
}

/// Subrecord to embed in the existing transaction Entry. It owns both original
/// and staged allocations until proof. It neither publishes logical pages nor
/// refunds runtime budgets. Unknown/unfenced drop deliberately leaks custody.
#[must_use = "retain in Entry until physical completion or explicit reap"]
pub struct PagePlacementCustody<B, F: PageCompletionFence> {
    owner: Identity,
    original: Vec<PagePhysicalPlacement<B>>,
    staged: Vec<PagePhysicalPlacement<B>>,
    fences: Option<[F; 3]>,
    generation: u64,
    state: State,
    canceled: bool,
    live_bytes: usize,
    slot_reuses: u64,
    active_submission: Option<Identity>,
}
impl<B, F: PageCompletionFence> PagePlacementCustody<B, F> {
    pub fn new(
        original: Vec<PagePhysicalPlacement<B>>,
        staged: Vec<PagePhysicalPlacement<B>>,
    ) -> Result<Self> {
        let all: Vec<_> = original.iter().chain(&staged).collect();
        let first = all
            .first()
            .ok_or_else(|| error("empty physical placement"))?;
        let mut grants = Vec::new();
        let mut live_bytes = 0usize;
        for p in &all {
            if p.allocation.0.domain != first.allocation.0.domain {
                return Err(error("foreign physical address space"));
            }
            if !grants.contains(&p.allocation.0.identity) {
                live_bytes = live_bytes
                    .checked_add(p.allocation.0.granted_bytes)
                    .ok_or_else(|| error("retained allocation bytes overflow"))?;
                grants.push(p.allocation.0.identity.clone());
            }
        }
        for (index, write) in staged.iter().enumerate() {
            if original
                .iter()
                .chain(&staged[..index])
                .any(|p| write.overlaps(p))
            {
                return Err(error(
                    "staged physical page aliases original or another write",
                ));
            }
        }
        Ok(Self {
            owner: Identity::new(),
            original,
            staged,
            fences: None,
            generation: 1,
            state: State::Idle,
            canceled: false,
            live_bytes,
            slot_reuses: 0,
            active_submission: None,
        })
    }

    pub fn metrics(&self) -> PagePlacementMetrics {
        PagePlacementMetrics {
            slot_reuses: self.slot_reuses,
            live_allocation_bytes: self.live_bytes,
            driver_freed_bytes: None,
        }
    }
    pub fn is_unknown(&self) -> bool {
        matches!(self.state, State::Unknown | State::Drained)
    }
    pub fn cancel(&mut self) {
        self.canceled = true;
    }

    /// Validate exact allocation incarnation, page/plane, generation and extent.
    /// Call before dispatch, together with the existing transaction/epoch checks.
    pub fn validate_chunk(&self, chunk: &PagePhysicalChunk) -> Result<()> {
        if matches!(
            self.state,
            State::Unknown | State::Drained | State::Released
        ) {
            return Err(error("placement quarantined or released"));
        }
        if self.original.iter().chain(&self.staged).any(|p| {
            p.chunk.identity == chunk.identity
                && p.chunk.page == chunk.page
                && p.chunk.plane == chunk.plane
                && p.chunk.byte_range.start <= chunk.byte_range.start
                && p.chunk.byte_range.end >= chunk.byte_range.end
                && chunk.byte_range.start < chunk.byte_range.end
        }) {
            Ok(())
        } else {
            Err(error("foreign or stale physical chunk"))
        }
    }

    /// Must precede any launch. Failure/unwind before fence installation keeps
    /// allocations charged and unreusable; dropping is not a completion signal.
    pub fn begin(&mut self) -> Result<PagePlacementSubmission> {
        if self.state != State::Idle || self.canceled {
            return Err(error("placement is not available for submission"));
        }
        self.state = State::Submitting;
        let submission = Identity::new();
        self.active_submission = Some(submission.clone());
        Ok(PagePlacementSubmission {
            owner: self.owner.clone(),
            submission,
            generation: self.generation,
        })
    }
    fn matches(&self, submission: &PagePlacementSubmission) -> Result<()> {
        if submission.owner != self.owner
            || submission.generation != self.generation
            || self.active_submission.as_ref() != Some(&submission.submission)
        {
            Err(error("foreign, stale or unbound physical submission"))
        } else {
            Ok(())
        }
    }

    fn validate_tokens(
        &self,
        tokens: &[PageCompletionToken<F>; 3],
        scope: CompletionScope,
    ) -> Result<()> {
        for (token, domain) in tokens.iter().zip([
            PageCompletionDomain::Producer,
            PageCompletionDomain::Dma,
            PageCompletionDomain::Consumer,
        ]) {
            self.matches(&token.submission)?;
            if token.domain != domain || token.scope != scope {
                return Err(error("completion token has wrong domain or drain scope"));
            }
        }
        Ok(())
    }

    /// Consume exact producer, DMA and final-consumer tokens, in that order.
    /// Construction of each token is a trusted unsafe adapter boundary. This
    /// method checks their opaque owner, generation, domain and submission scope.
    pub fn record_fences(
        &mut self,
        submission: &PagePlacementSubmission,
        tokens: [PageCompletionToken<F>; 3],
    ) -> Result<()> {
        self.matches(submission)?;
        if self.state != State::Submitting {
            return Err(error("fences already bound or quarantined"));
        }
        self.validate_tokens(&tokens, CompletionScope::Submission)?;
        self.fences = Some(tokens.map(|token| token.fence));
        self.state = State::Pending;
        Ok(())
    }
    pub fn mark_unknown(&mut self, submission: &PagePlacementSubmission) -> Result<()> {
        self.matches(submission)?;
        if !matches!(
            self.state,
            State::Submitting | State::Pending | State::Unknown
        ) {
            return Err(error("no physical submission to quarantine"));
        }
        self.state = State::Unknown;
        Ok(())
    }
    pub fn poll(
        &mut self,
        submission: &PagePlacementSubmission,
    ) -> Result<Option<PagePlacementProof>> {
        self.matches(submission)?;
        if self.state != State::Pending {
            return Err(error(
                "placement is unfenced, quarantined or already proved",
            ));
        }
        // Remain Unknown even if a fence probe unwinds.
        self.state = State::Unknown;
        let complete = Self::complete(self.fences.as_ref().expect("bound fences"))?;
        if complete {
            self.state = State::Ready;
            Ok(Some(PagePlacementProof(submission.clone())))
        } else {
            self.state = State::Pending;
            Ok(None)
        }
    }
    fn complete(fences: &[F; 3]) -> Result<bool> {
        let mut complete = true;
        for fence in fences {
            complete &= fence.is_complete()?;
        }
        Ok(complete)
    }

    /// Explicit physical drain permits release, NOT reuse or logical recovery.
    /// Unknown stays sticky (Drained) even after all three domains complete.
    /// Submission/late-ACK tokens cannot substitute for full-drain tokens, and
    /// partial drains retain both allocations and old fences. The existing Entry
    /// must independently satisfy its terminal abort/retirement obligations.
    pub fn reap(
        &mut self,
        submission: &PagePlacementSubmission,
        tokens: [PageCompletionToken<F>; 3],
    ) -> Result<Option<PagePlacementProof>> {
        self.matches(submission)?;
        if !matches!(self.state, State::Unknown | State::Submitting) {
            return Err(error("reap requires unknown or unfenced submission"));
        }
        self.validate_tokens(&tokens, CompletionScope::FullDrain)?;
        self.state = State::Unknown;
        let drains = tokens.map(|token| token.fence);
        if !Self::complete(&drains)? {
            return Ok(None);
        }
        self.fences = Some(drains);
        self.state = State::Drained;
        Ok(Some(PagePlacementProof(submission.clone())))
    }
    fn check_proof(&self, proof: &PagePlacementProof) -> Result<()> {
        self.matches(&proof.0)?;
        if !matches!(self.state, State::Ready | State::Drained) {
            return Err(error("physical proof is not current"));
        }
        Ok(())
    }

    /// Release this Entry's grant references, not driver memory or logical pins.
    pub fn release(&mut self, proof: PagePlacementProof) -> Result<()> {
        self.check_proof(&proof)?;
        self.original.clear();
        self.staged.clear();
        self.fences = None;
        self.live_bytes = 0;
        self.active_submission = None;
        self.state = State::Released;
        Ok(())
    }

    /// Reuse exactly these physical extents after proof. Logical page/plane
    /// bindings stay unchanged; publishing/remapping remains the existing Entry's
    /// job. A new generation invalidates all old chunks, submissions and proofs.
    /// No allocation bytes are reclaimed by this operation.
    pub fn reuse(&mut self, proof: PagePlacementProof) -> Result<()> {
        self.check_proof(&proof)?;
        if self.canceled || self.state == State::Drained {
            return Err(error(
                "canceled or drained-Unknown placement cannot be reused",
            ));
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| error("placement generation exhausted"))?;
        let slots = self
            .original
            .iter()
            .chain(&self.staged)
            .filter(|p| p.allocation.0.kind == PageAllocationKind::FixedPlane)
            .count();
        let slot_reuses = self
            .slot_reuses
            .checked_add(slots as u64)
            .ok_or_else(|| error("slot reuse counter exhausted"))?;
        for p in self.original.iter_mut().chain(&mut self.staged) {
            p.chunk.identity.generation = generation;
        }
        self.generation = generation;
        self.slot_reuses = slot_reuses;
        self.fences = None;
        self.active_submission = None;
        self.state = State::Idle;
        Ok(())
    }

    pub fn original(&self) -> &[PagePhysicalPlacement<B>] {
        &self.original
    }
    pub fn staged(&self) -> &[PagePhysicalPlacement<B>] {
        &self.staged
    }
}
impl<B, F: PageCompletionFence> Drop for PagePlacementCustody<B, F> {
    fn drop(&mut self) {
        if matches!(
            self.state,
            State::Submitting | State::Pending | State::Unknown
        ) {
            std::mem::forget(std::mem::take(&mut self.original));
            std::mem::forget(std::mem::take(&mut self.staged));
            std::mem::forget(self.fences.take());
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/physical_placement.rs"]
mod tests;
