//! Layered, model-independent transformer forward execution.

use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;

use ferrule_common::{Error, Result};

use crate::decoder::{
    DecoderCancelProgress, DecoderForwardExecutor, DecoderForwardProgress,
    DecoderForwardResumeProgress, DecoderLogits, DecoderTerminalGuard, DecoderTransactionContext,
    DecoderWait, PackedDecoderBatch,
};
use crate::execution::{OwnedArenaCheckout, PersistentArenaPool, PersistentArenaPoolStats};

/// One residual/connection strategy wrapped around one physical block.
#[derive(Debug, Clone)]
pub struct Connected<C, B> {
    connection: C,
    block: B,
}

impl<C, B> Connected<C, B> {
    pub const fn new(connection: C, block: B) -> Self {
        Self { connection, block }
    }

    pub const fn connection(&self) -> &C {
        &self.connection
    }

    pub const fn block(&self) -> &B {
        &self.block
    }
}

/// Exact attention/feed-forward composition for one transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayer<Attention, FeedForward> {
    index: usize,
    attention: Attention,
    feed_forward: FeedForward,
}

impl<Attention, FeedForward> TransformerLayer<Attention, FeedForward> {
    pub const fn new(index: usize, attention: Attention, feed_forward: FeedForward) -> Self {
        Self {
            index,
            attention,
            feed_forward,
        }
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn attention(&self) -> &Attention {
        &self.attention
    }

    pub const fn feed_forward(&self) -> &FeedForward {
        &self.feed_forward
    }
}

/// Ordinary additive residual connection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AddResidual;

/// Physical layer mode. MTP uses the same layer composition with proposal
/// metadata instead of maintaining a second transformer implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerMode {
    Target,
    Proposal {
        stage: usize,
        sequence_tokens: usize,
    },
}

/// Typed request passed to one concrete transformer layer.
#[derive(Debug)]
pub enum LayerRequest<'a, S, K> {
    Target {
        batch: &'a PackedDecoderBatch,
        states: &'a mut [S],
        kv: &'a mut K,
    },
    Proposal {
        stage: usize,
        sequence_tokens: usize,
        token_ids: &'a [u32],
        states: &'a mut [S],
        kv: &'a mut K,
    },
}

impl<S, K> LayerRequest<'_, S, K> {
    pub const fn mode(&self) -> LayerMode {
        match self {
            Self::Target { .. } => LayerMode::Target,
            Self::Proposal {
                stage,
                sequence_tokens,
                ..
            } => LayerMode::Proposal {
                stage: *stage,
                sequence_tokens: *sequence_tokens,
            },
        }
    }
}

/// A typed asynchronous state and its optional externally visible wait edge.
/// `wait == None` means the physical operation may be polled immediately.
#[derive(Debug)]
pub struct Pending<P> {
    state: P,
    wait: Option<DecoderWait>,
}

impl<P> Pending<P> {
    pub const fn physical(state: P) -> Self {
        Self { state, wait: None }
    }

    pub const fn waiting(state: P, wait: DecoderWait) -> Self {
        Self {
            state,
            wait: Some(wait),
        }
    }

    pub const fn state(&self) -> &P {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut P {
        &mut self.state
    }

    pub const fn wait(&self) -> Option<&DecoderWait> {
        self.wait.as_ref()
    }

    pub fn into_state(self) -> P {
        self.state
    }
}

/// Initial submission result for embedding, layer, or output-head work.
#[derive(Debug)]
pub enum Step<T, P> {
    Complete(T),
    Pending(Pending<P>),
}

/// Poll result for already-submitted physical work.
#[derive(Debug)]
pub enum Poll<T, P> {
    Ready(T),
    Pending(Pending<P>),
}

/// Model-independent output pipeline composition.
#[derive(Debug, Clone)]
pub struct OutputPipeline<Reduction, Norm, Head> {
    reduction: Reduction,
    norm: Norm,
    head: Head,
}

impl<Reduction, Norm, Head> OutputPipeline<Reduction, Norm, Head> {
    pub const fn new(reduction: Reduction, norm: Norm, head: Head) -> Self {
        Self {
            reduction,
            norm,
            head,
        }
    }

    pub const fn reduction(&self) -> &Reduction {
        &self.reduction
    }

    pub const fn norm(&self) -> &Norm {
        &self.norm
    }

    pub const fn head(&self) -> &Head {
        &self.head
    }
}

/// Output connection strategy for ordinary one-stream decoders.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoReduction;

/// Output connection strategy for multi-stream HyperConnection decoders.
#[derive(Debug, Clone)]
pub struct HyperReduction<H> {
    head: H,
}

impl<H> HyperReduction<H> {
    pub const fn new(head: H) -> Self {
        Self { head }
    }

    pub const fn head(&self) -> &H {
        &self.head
    }
}

/// No-op target tap policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoPostLayerTap;

/// MTP target hidden-state tap policy. Physical tap buffers remain backend typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpTap {
    target_layers: Box<[usize]>,
}

impl MtpTap {
    pub fn new(target_layers: impl Into<Box<[usize]>>) -> Result<Self> {
        let target_layers = target_layers.into();
        if target_layers.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(transformer_error(
                "MTP target tap layers must be strictly increasing",
            ));
        }
        Ok(Self { target_layers })
    }

    pub fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    pub fn captures(&self, layer: usize) -> bool {
        self.target_layers.binary_search(&layer).is_ok()
    }
}

/// Fully prepared transformer components selected by one model recipe/adapter.
#[derive(Debug)]
pub struct PreparedTransformer<Embedding, Layer, Output, Tap> {
    embedding: Embedding,
    layers: Box<[Layer]>,
    output: Output,
    post_layer_tap: Tap,
}

impl<Embedding, Layer, Output, Tap> PreparedTransformer<Embedding, Layer, Output, Tap> {
    pub fn new(
        embedding: Embedding,
        layers: impl Into<Box<[Layer]>>,
        output: Output,
        post_layer_tap: Tap,
    ) -> Self {
        Self {
            embedding,
            layers: layers.into(),
            output,
            post_layer_tap,
        }
    }

    pub const fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub const fn output(&self) -> &Output {
        &self.output
    }

    pub const fn post_layer_tap(&self) -> &Tap {
        &self.post_layer_tap
    }
}

/// Shared owned arena pool. The pool is private to one executor, while an active
/// continuation may retain its arena without borrowing the executor.
#[derive(Debug)]
struct OwnedArenaPool<K, A> {
    inner: Rc<RefCell<PersistentArenaPool<K, A>>>,
}

impl<K, A> Default for OwnedArenaPool<K, A>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self {
            inner: Rc::new(RefCell::new(PersistentArenaPool::new())),
        }
    }
}

impl<K, A> OwnedArenaPool<K, A>
where
    K: Eq + Hash,
{
    fn checkout(&self, key: K, build: impl FnOnce() -> Result<A>) -> Result<OwnedArenaLease<K, A>> {
        let checkout = self.inner.borrow_mut().checkout(key, build)?;
        Ok(OwnedArenaLease {
            pool: Rc::clone(&self.inner),
            checkout: Some(checkout),
        })
    }

    fn clear(&self) {
        self.inner.borrow_mut().clear();
    }

    fn stats(&self) -> PersistentArenaPoolStats {
        self.inner.borrow().stats()
    }
}

/// RAII custody for an owned arena checkout. Every terminal path returns the
/// arena to its exact-key bucket; models never perform `Option::take` cleanup.
pub struct OwnedArenaLease<K, A>
where
    K: Eq + Hash,
{
    pool: Rc<RefCell<PersistentArenaPool<K, A>>>,
    checkout: Option<OwnedArenaCheckout<K, A>>,
}

impl<K, A> Debug for OwnedArenaLease<K, A>
where
    K: Debug + Eq + Hash,
    A: Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnedArenaLease")
            .field("checkout", &self.checkout)
            .finish()
    }
}

impl<K, A> OwnedArenaLease<K, A>
where
    K: Eq + Hash,
{
    pub fn get(&self) -> &A {
        self.checkout
            .as_ref()
            .expect("owned arena lease is present")
            .get()
    }

    pub fn get_mut(&mut self) -> &mut A {
        self.checkout
            .as_mut()
            .expect("owned arena lease is present")
            .get_mut()
    }
}

impl<K, A> Drop for OwnedArenaLease<K, A>
where
    K: Eq + Hash,
{
    fn drop(&mut self) {
        if let Some(checkout) = self.checkout.take() {
            self.pool
                .borrow_mut()
                .checkin(checkout)
                .expect("arena checkin is infallible");
        }
    }
}

/// Exact typed boundary implemented by one concrete transformer composition.
/// The generic executor owns the only embedding/layer/output cursor and all
/// wait, resume, cancellation, quiescence, and arena lifecycle transitions.
pub trait TransformerModule: Sized {
    type State;
    type KvView;
    type Hidden;
    type ArenaKey: Debug + Eq + Hash;
    type Arena: Debug;
    type EmbeddingPending: Debug;
    type LayerPending: Debug;
    type OutputPending: Debug;
    type Quiescence: Debug;
    type Event: Debug;
    type TerminalGuard: DecoderTerminalGuard;

    /// Number of target layers driven per forward. Prepared composition
    /// details stay module-private; the executor only sequences layers.
    fn layer_count(&self) -> usize;

    fn validate(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
    ) -> Result<()>;

    fn arena_key(&self, batch: &PackedDecoderBatch) -> Result<Self::ArenaKey>;
    fn build_arena(&mut self, key: &Self::ArenaKey) -> Result<Self::Arena>;

    fn submit_embedding(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
        arena: &mut Self::Arena,
    ) -> Result<Step<Self::Hidden, Self::EmbeddingPending>>;

    fn poll_embedding(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
        arena: &mut Self::Arena,
        pending: &mut Self::EmbeddingPending,
    ) -> Result<Poll<Self::Hidden, Self::EmbeddingPending>>;

    fn start_layer(
        &mut self,
        context: &DecoderTransactionContext,
        layer: usize,
        request: LayerRequest<'_, Self::State, Self::KvView>,
        hidden: &mut Self::Hidden,
        arena: &mut Self::Arena,
    ) -> Result<Step<Vec<Self::Event>, Self::LayerPending>>;

    fn poll_layer(
        &mut self,
        context: &DecoderTransactionContext,
        layer: usize,
        request: LayerRequest<'_, Self::State, Self::KvView>,
        hidden: &mut Self::Hidden,
        arena: &mut Self::Arena,
        pending: &mut Self::LayerPending,
    ) -> Result<Poll<Vec<Self::Event>, Self::LayerPending>>;

    fn post_layer(
        &mut self,
        layer: usize,
        batch: &PackedDecoderBatch,
        hidden: &Self::Hidden,
        arena: &mut Self::Arena,
    ) -> Result<()>;

    fn submit_output(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
        hidden: &Self::Hidden,
        arena: &mut Self::Arena,
    ) -> Result<Step<DecoderLogits, Self::OutputPending>>;

    fn poll_output(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
        hidden: &Self::Hidden,
        arena: &mut Self::Arena,
        pending: &mut Self::OutputPending,
    ) -> Result<Poll<DecoderLogits, Self::OutputPending>>;

    fn poll_embedding_cancel(&mut self, pending: &mut Self::EmbeddingPending) -> Result<bool>;
    fn poll_layer_cancel(&mut self, layer: usize, pending: &mut Self::LayerPending)
    -> Result<bool>;
    fn poll_output_cancel(&mut self, pending: &mut Self::OutputPending) -> Result<bool>;
    fn cancel_embedding(&mut self, pending: Self::EmbeddingPending) -> Result<()>;
    fn cancel_layer(&mut self, layer: usize, pending: Self::LayerPending) -> Result<()>;
    fn cancel_output(&mut self, pending: Self::OutputPending) -> Result<()>;
    fn begin_quiescence(&mut self) -> Result<Self::Quiescence>;
    fn poll_quiescence(&mut self, quiescence: &mut Self::Quiescence) -> Result<bool>;

    fn finish(
        &mut self,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
        arena: &mut Self::Arena,
        events: Vec<Self::Event>,
    ) -> Result<Self::TerminalGuard>;

    fn abort(
        &mut self,
        batch: &PackedDecoderBatch,
        states: &mut [Self::State],
        kv: &mut Self::KvView,
        arena: &mut Self::Arena,
    ) -> Result<()>;

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

enum PhysicalPhase<M>
where
    M: TransformerModule,
{
    EmbeddingReady,
    EmbeddingPending(Pending<M::EmbeddingPending>),
    LayerReady,
    LayerPending(Pending<M::LayerPending>),
    OutputReady,
    OutputPending(Pending<M::OutputPending>),
    Cancelling(M::Quiescence),
    Finished,
}

struct TransformerForwardState<M>
where
    M: TransformerModule,
{
    arena: OwnedArenaLease<M::ArenaKey, M::Arena>,
    hidden: Option<M::Hidden>,
    layer_cursor: usize,
    phase: PhysicalPhase<M>,
    events: Vec<M::Event>,
    failed: bool,
}

fn phase_name<M>(phase: &PhysicalPhase<M>) -> &'static str
where
    M: TransformerModule,
{
    match phase {
        PhysicalPhase::EmbeddingReady => "embedding-ready",
        PhysicalPhase::EmbeddingPending(_) => "embedding-pending",
        PhysicalPhase::LayerReady => "layer-ready",
        PhysicalPhase::LayerPending(_) => "layer-pending",
        PhysicalPhase::OutputReady => "output-ready",
        PhysicalPhase::OutputPending(_) => "output-pending",
        PhysicalPhase::Cancelling(_) => "cancelling",
        PhysicalPhase::Finished => "finished",
    }
}

/// Typed continuation retained by the decoder transaction shell.
pub struct TransformerContinuation<M>
where
    M: TransformerModule,
{
    state: TransformerForwardState<M>,
}

impl<M> Debug for TransformerContinuation<M>
where
    M: TransformerModule,
    M::Hidden: Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TransformerContinuation")
            .field("layer_cursor", &self.state.layer_cursor)
            .field("phase", &phase_name(&self.state.phase))
            .field("failed", &self.state.failed)
            .finish_non_exhaustive()
    }
}

/// The sole decoder-forward layer cursor and physical phase machine.
pub struct TransformerForwardExecutor<M>
where
    M: TransformerModule,
{
    module: M,
    arenas: OwnedArenaPool<M::ArenaKey, M::Arena>,
}

impl<M> TransformerForwardExecutor<M>
where
    M: TransformerModule,
{
    pub fn new(module: M) -> Self {
        Self {
            module,
            arenas: OwnedArenaPool::default(),
        }
    }

    pub const fn module(&self) -> &M {
        &self.module
    }

    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }

    pub fn arena_stats(&self) -> PersistentArenaPoolStats {
        self.arenas.stats()
    }

    fn begin_state(&mut self, batch: &PackedDecoderBatch) -> Result<TransformerForwardState<M>> {
        let key = self.module.arena_key(batch)?;
        let arena = self.arenas.checkout(key, || {
            let key = self.module.arena_key(batch)?;
            self.module.build_arena(&key)
        })?;
        Ok(TransformerForwardState {
            arena,
            hidden: None,
            layer_cursor: 0,
            phase: PhysicalPhase::EmbeddingReady,
            events: Vec::new(),
            failed: false,
        })
    }

    fn wait_or_continue<P>(pending: &Pending<P>) -> Option<DecoderWait> {
        pending.wait().cloned()
    }

    fn drive(
        &mut self,
        context: &DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [M::State],
        kv: &mut M::KvView,
        state: &mut TransformerForwardState<M>,
        mut woke: bool,
    ) -> Result<DriveProgress<M::TerminalGuard>> {
        if state.failed {
            return Err(transformer_error(
                "forward previously failed and must be cancelled",
            ));
        }
        loop {
            let phase = std::mem::replace(&mut state.phase, PhysicalPhase::Finished);
            match phase {
                PhysicalPhase::EmbeddingReady => match self.module.submit_embedding(
                    context,
                    batch,
                    states,
                    kv,
                    state.arena.get_mut(),
                )? {
                    Step::Complete(hidden) => {
                        state.hidden = Some(hidden);
                        state.phase = PhysicalPhase::LayerReady;
                    }
                    Step::Pending(pending) => {
                        let wait = Self::wait_or_continue(&pending);
                        state.phase = PhysicalPhase::EmbeddingPending(pending);
                        if let Some(wait) = wait {
                            return Ok(DriveProgress::Waiting(wait));
                        }
                    }
                },
                PhysicalPhase::EmbeddingPending(mut pending) => {
                    if pending.wait().is_some() && !woke {
                        state.phase = PhysicalPhase::EmbeddingPending(pending);
                        return Err(transformer_error(
                            "embedding physical work was polled without a wake",
                        ));
                    }
                    woke = false;
                    match self.module.poll_embedding(
                        context,
                        batch,
                        states,
                        kv,
                        state.arena.get_mut(),
                        pending.state_mut(),
                    )? {
                        Poll::Ready(hidden) => {
                            state.hidden = Some(hidden);
                            state.phase = PhysicalPhase::LayerReady;
                        }
                        Poll::Pending(next) => {
                            let wait = Self::wait_or_continue(&next);
                            state.phase = PhysicalPhase::EmbeddingPending(next);
                            if let Some(wait) = wait {
                                return Ok(DriveProgress::Waiting(wait));
                            }
                        }
                    }
                }
                PhysicalPhase::LayerReady => {
                    if state.layer_cursor == self.module.layer_count() {
                        state.phase = PhysicalPhase::OutputReady;
                        continue;
                    }
                    let hidden = state.hidden.as_mut().ok_or_else(|| {
                        transformer_error("layer execution has no embedded hidden state")
                    })?;
                    let request = LayerRequest::Target { batch, states, kv };
                    match self.module.start_layer(
                        context,
                        state.layer_cursor,
                        request,
                        hidden,
                        state.arena.get_mut(),
                    )? {
                        Step::Complete(events) => {
                            state.events.extend(events);
                            self.module.post_layer(
                                state.layer_cursor,
                                batch,
                                hidden,
                                state.arena.get_mut(),
                            )?;
                            state.layer_cursor += 1;
                            state.phase = PhysicalPhase::LayerReady;
                        }
                        Step::Pending(pending) => {
                            let wait = Self::wait_or_continue(&pending);
                            state.phase = PhysicalPhase::LayerPending(pending);
                            if let Some(wait) = wait {
                                return Ok(DriveProgress::Waiting(wait));
                            }
                        }
                    }
                }
                PhysicalPhase::LayerPending(mut pending) => {
                    if pending.wait().is_some() && !woke {
                        state.phase = PhysicalPhase::LayerPending(pending);
                        return Err(transformer_error(
                            "layer physical work was polled without a wake",
                        ));
                    }
                    woke = false;
                    let hidden = state.hidden.as_mut().ok_or_else(|| {
                        transformer_error("resumed layer has no embedded hidden state")
                    })?;
                    let request = LayerRequest::Target { batch, states, kv };
                    match self.module.poll_layer(
                        context,
                        state.layer_cursor,
                        request,
                        hidden,
                        state.arena.get_mut(),
                        pending.state_mut(),
                    )? {
                        Poll::Ready(events) => {
                            state.events.extend(events);
                            self.module.post_layer(
                                state.layer_cursor,
                                batch,
                                hidden,
                                state.arena.get_mut(),
                            )?;
                            state.layer_cursor += 1;
                            state.phase = PhysicalPhase::LayerReady;
                        }
                        Poll::Pending(next) => {
                            let wait = Self::wait_or_continue(&next);
                            state.phase = PhysicalPhase::LayerPending(next);
                            if let Some(wait) = wait {
                                return Ok(DriveProgress::Waiting(wait));
                            }
                        }
                    }
                }
                PhysicalPhase::OutputReady => {
                    let hidden = state
                        .hidden
                        .as_ref()
                        .ok_or_else(|| transformer_error("output pipeline has no hidden state"))?;
                    match self.module.submit_output(
                        context,
                        batch,
                        states,
                        kv,
                        hidden,
                        state.arena.get_mut(),
                    )? {
                        Step::Complete(logits) => {
                            let terminal_guard = self.module.finish(
                                batch,
                                states,
                                kv,
                                state.arena.get_mut(),
                                std::mem::take(&mut state.events),
                            )?;
                            state.phase = PhysicalPhase::Finished;
                            return Ok(DriveProgress::Complete {
                                logits,
                                terminal_guard,
                            });
                        }
                        Step::Pending(pending) => {
                            let wait = Self::wait_or_continue(&pending);
                            state.phase = PhysicalPhase::OutputPending(pending);
                            if let Some(wait) = wait {
                                return Ok(DriveProgress::Waiting(wait));
                            }
                        }
                    }
                }
                PhysicalPhase::OutputPending(mut pending) => {
                    if pending.wait().is_some() && !woke {
                        state.phase = PhysicalPhase::OutputPending(pending);
                        return Err(transformer_error(
                            "output-head physical work was polled without a wake",
                        ));
                    }
                    woke = false;
                    let hidden = state.hidden.as_ref().ok_or_else(|| {
                        transformer_error("polled output pipeline has no hidden state")
                    })?;
                    match self.module.poll_output(
                        context,
                        batch,
                        states,
                        kv,
                        hidden,
                        state.arena.get_mut(),
                        pending.state_mut(),
                    )? {
                        Poll::Ready(logits) => {
                            let terminal_guard = self.module.finish(
                                batch,
                                states,
                                kv,
                                state.arena.get_mut(),
                                std::mem::take(&mut state.events),
                            )?;
                            state.phase = PhysicalPhase::Finished;
                            return Ok(DriveProgress::Complete {
                                logits,
                                terminal_guard,
                            });
                        }
                        Poll::Pending(next) => {
                            let wait = Self::wait_or_continue(&next);
                            state.phase = PhysicalPhase::OutputPending(next);
                            if let Some(wait) = wait {
                                return Ok(DriveProgress::Waiting(wait));
                            }
                        }
                    }
                }
                PhysicalPhase::Cancelling(quiescence) => {
                    state.phase = PhysicalPhase::Cancelling(quiescence);
                    return Err(transformer_error(
                        "cancelled forward cannot resume normal execution",
                    ));
                }
                PhysicalPhase::Finished => {
                    state.phase = PhysicalPhase::Finished;
                    return Err(transformer_error("forward is already complete"));
                }
            }
        }
    }

    fn poll_cancel(
        &mut self,
        batch: &PackedDecoderBatch,
        states: &mut [M::State],
        kv: &mut M::KvView,
        state: &mut TransformerForwardState<M>,
    ) -> Result<DecoderCancelProgress> {
        let phase = std::mem::replace(&mut state.phase, PhysicalPhase::Finished);
        let ready = match phase {
            PhysicalPhase::EmbeddingPending(mut pending) => {
                if !self.module.poll_embedding_cancel(pending.state_mut())? {
                    state.phase = PhysicalPhase::EmbeddingPending(pending);
                    return Ok(DecoderCancelProgress::Waiting);
                }
                self.module.cancel_embedding(pending.into_state())?;
                true
            }
            PhysicalPhase::LayerPending(mut pending) => {
                if !self
                    .module
                    .poll_layer_cancel(state.layer_cursor, pending.state_mut())?
                {
                    state.phase = PhysicalPhase::LayerPending(pending);
                    return Ok(DecoderCancelProgress::Waiting);
                }
                self.module
                    .cancel_layer(state.layer_cursor, pending.into_state())?;
                true
            }
            PhysicalPhase::OutputPending(mut pending) => {
                if !self.module.poll_output_cancel(pending.state_mut())? {
                    state.phase = PhysicalPhase::OutputPending(pending);
                    return Ok(DecoderCancelProgress::Waiting);
                }
                self.module.cancel_output(pending.into_state())?;
                true
            }
            PhysicalPhase::Cancelling(mut quiescence) => {
                if !self.module.poll_quiescence(&mut quiescence)? {
                    state.phase = PhysicalPhase::Cancelling(quiescence);
                    return Ok(DecoderCancelProgress::Waiting);
                }
                true
            }
            PhysicalPhase::EmbeddingReady
            | PhysicalPhase::LayerReady
            | PhysicalPhase::OutputReady => true,
            PhysicalPhase::Finished => {
                state.phase = PhysicalPhase::Finished;
                return Ok(DecoderCancelProgress::Complete);
            }
        };
        if ready && !matches!(state.phase, PhysicalPhase::Cancelling(_)) {
            let mut quiescence = self.module.begin_quiescence()?;
            if !self.module.poll_quiescence(&mut quiescence)? {
                state.phase = PhysicalPhase::Cancelling(quiescence);
                return Ok(DecoderCancelProgress::Waiting);
            }
        }
        self.module
            .abort(batch, states, kv, state.arena.get_mut())?;
        state.events.clear();
        state.phase = PhysicalPhase::Finished;
        Ok(DecoderCancelProgress::Complete)
    }
}

#[derive(Debug)]
enum DriveProgress<G> {
    Waiting(DecoderWait),
    Complete {
        logits: DecoderLogits,
        terminal_guard: G,
    },
}

impl<M> DecoderForwardExecutor<M::State, M::KvView> for TransformerForwardExecutor<M>
where
    M: TransformerModule,
    M::Hidden: Debug,
{
    type Continuation = TransformerContinuation<M>;
    type TerminalGuard = M::TerminalGuard;

    fn start_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [M::State],
        kv: &mut M::KvView,
    ) -> Result<DecoderForwardProgress<Self::Continuation, Self::TerminalGuard>> {
        if let Err(error) = self.module.validate(context, batch, states, kv) {
            return Ok(DecoderForwardProgress::FailedQuiescent(error));
        }
        let mut state = match self.begin_state(batch) {
            Ok(state) => state,
            Err(error) => return Ok(DecoderForwardProgress::FailedQuiescent(error)),
        };
        match self.drive(context, batch, states, kv, &mut state, false) {
            Ok(DriveProgress::Waiting(wait)) => Ok(DecoderForwardProgress::Waiting {
                continuation: TransformerContinuation { state },
                wait,
            }),
            Ok(DriveProgress::Complete {
                logits,
                terminal_guard,
            }) => Ok(DecoderForwardProgress::Complete {
                logits,
                terminal_guard,
            }),
            Err(error) => {
                state.failed = true;
                Ok(DecoderForwardProgress::FailedActive {
                    continuation: TransformerContinuation { state },
                    error,
                })
            }
        }
    }

    fn resume_forward(
        &mut self,
        context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [M::State],
        kv: &mut M::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderForwardResumeProgress<Self::TerminalGuard>> {
        self.module.validate(context, batch, states, kv)?;
        match self.drive(context, batch, states, kv, &mut continuation.state, true) {
            Ok(DriveProgress::Waiting(wait)) => Ok(DecoderForwardResumeProgress::Waiting(wait)),
            Ok(DriveProgress::Complete {
                logits,
                terminal_guard,
            }) => Ok(DecoderForwardResumeProgress::Complete {
                logits,
                terminal_guard,
            }),
            Err(error) => {
                continuation.state.failed = true;
                Ok(DecoderForwardResumeProgress::FailedActive(error))
            }
        }
    }

    fn cancel_forward(
        &mut self,
        _context: &mut DecoderTransactionContext,
        batch: &PackedDecoderBatch,
        states: &mut [M::State],
        kv: &mut M::KvView,
        continuation: &mut Self::Continuation,
    ) -> Result<DecoderCancelProgress> {
        self.poll_cancel(batch, states, kv, &mut continuation.state)
    }

    fn shutdown(&mut self) -> Result<()> {
        self.arenas.clear();
        self.module.shutdown()
    }
}

fn transformer_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: format!("transformer forward: {}", message.into()),
    }
}
