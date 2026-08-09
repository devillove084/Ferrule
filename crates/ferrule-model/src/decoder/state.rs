use super::{DecoderSequence, DecoderSequenceCheckout, DecoderSequenceLifecycle};
use crate::execution::{SequenceStateCore, SequenceTopologyId};
use crate::runner::SequenceStateReleaseError;
use ferrule_common::{Error, Result};
use std::marker::PhantomData;
/// Model-owned per-sequence state with generic attachment and KV metadata.
///
/// A topology identifies one logical sequence. Transaction working copies retain
/// both topology and generation, while logical forks receive a fresh topology and
/// an invalidating generation bump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderSequenceState<Attachment, KvState> {
    core: SequenceStateCore,
    topology_id: SequenceTopologyId,
    attachment: Attachment,
    kv_state: KvState,
}
impl<Attachment, KvState> DecoderSequenceState<Attachment, KvState> {
    pub fn new(attachment: Attachment, kv_state: KvState) -> Self {
        Self {
            core: SequenceStateCore::new(),
            topology_id: SequenceTopologyId::take(),
            attachment,
            kv_state,
        }
    }
    pub fn with_position(position: usize, attachment: Attachment, kv_state: KvState) -> Self {
        Self {
            core: SequenceStateCore::with_position(position),
            topology_id: SequenceTopologyId::take(),
            attachment,
            kv_state,
        }
    }
    pub const fn core(&self) -> &SequenceStateCore {
        &self.core
    }
    pub fn core_mut(&mut self) -> &mut SequenceStateCore {
        &mut self.core
    }
    pub const fn topology_id(&self) -> SequenceTopologyId {
        self.topology_id
    }
    pub const fn attachment(&self) -> &Attachment {
        &self.attachment
    }
    pub fn attachment_mut(&mut self) -> &mut Attachment {
        &mut self.attachment
    }
    pub const fn kv_state(&self) -> &KvState {
        &self.kv_state
    }
    pub fn kv_state_mut(&mut self) -> &mut KvState {
        &mut self.kv_state
    }
    pub fn into_parts(self) -> (SequenceStateCore, SequenceTopologyId, Attachment, KvState) {
        (self.core, self.topology_id, self.attachment, self.kv_state)
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn from_parts(
        core: SequenceStateCore,
        topology_id: SequenceTopologyId,
        attachment: Attachment,
        kv_state: KvState,
    ) -> Self {
        Self {
            core,
            topology_id,
            attachment,
            kv_state,
        }
    }
    /// Produces a transaction-local copy of committed state.
    ///
    /// The exact topology, cursor, and generation are retained so reservations
    /// remain bound to the committed source until publication.
    pub fn transaction_working_copy(&self) -> Result<Self>
    where
        Attachment: Clone,
        KvState: Clone,
    {
        self.core.begin_step()?;
        Ok(self.clone())
    }
    /// Produces an independent logical branch at the same committed cursor.
    ///
    /// The branch receives a fresh topology. Its core generation is advanced so
    /// bindings captured from the source cannot be replayed against the branch.
    pub fn logical_fork(&self) -> Result<Self>
    where
        Attachment: Clone,
        KvState: Clone,
    {
        Ok(Self {
            core: self.core.forked()?,
            topology_id: SequenceTopologyId::take(),
            attachment: self.attachment.clone(),
            kv_state: self.kv_state.clone(),
        })
    }
    /// Releases model-owned attachment resources without losing state on preflight
    /// failure. Once preflight succeeds, `DecoderSequenceAttachment::release` is infallible.
    pub fn try_release_attachment(
        self,
    ) -> std::result::Result<KvState, DecoderSequenceReleaseError<Attachment, KvState>>
    where
        Attachment: DecoderSequenceAttachment,
    {
        let release = match self.attachment.preflight_release() {
            Ok(release) => release,
            Err(source) => {
                return Err(DecoderSequenceReleaseError {
                    source,
                    state: self,
                });
            }
        };
        let Self {
            core: _,
            topology_id: _,
            attachment,
            kv_state,
        } = self;
        attachment.release(release);
        Ok(kv_state)
    }
}
impl<Attachment, KvState> DecoderSequence for DecoderSequenceState<Attachment, KvState> {
    fn topology_id(&self) -> SequenceTopologyId {
        self.topology_id
    }
    fn core(&self) -> &SequenceStateCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut SequenceStateCore {
        &mut self.core
    }
}
/// Release lifecycle for model-specific sequence attachments.
///
/// All fallible work belongs in `preflight_release`. Implementations must not
/// mutate externally visible state during preflight. `release` is deliberately
/// infallible, making custody transfer explicit and exactly once.
pub trait DecoderSequenceAttachment: Sized {
    type Release;
    fn preflight_release(&self) -> Result<Self::Release>;
    fn release(self, release: Self::Release);
}
/// Default lifecycle for cloneable standard decoder attachments.
///
/// The generic transaction shell itself has no clone bound; this adapter supplies
/// clone-based lifecycle behavior for Qwen and other ordinary CPU decoders.
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardSequenceLifecycle<Attachment, KvState> {
    marker: PhantomData<fn() -> (Attachment, KvState)>,
}
impl<Attachment, KvState> StandardSequenceLifecycle<Attachment, KvState> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}
impl<Attachment, KvState> DecoderSequenceLifecycle<DecoderSequenceState<Attachment, KvState>>
    for StandardSequenceLifecycle<Attachment, KvState>
where
    Attachment: Clone + Default + DecoderSequenceAttachment,
    KvState: Clone + Default,
{
    fn create(&mut self) -> Result<DecoderSequenceState<Attachment, KvState>> {
        Ok(DecoderSequenceState::new(
            Attachment::default(),
            KvState::default(),
        ))
    }
    fn checkout(
        &mut self,
        _request: DecoderSequenceCheckout,
        source: &DecoderSequenceState<Attachment, KvState>,
    ) -> Result<DecoderSequenceState<Attachment, KvState>> {
        source.transaction_working_copy()
    }
    fn logical_fork(
        &mut self,
        source: &DecoderSequenceState<Attachment, KvState>,
        expected_position: usize,
    ) -> Result<DecoderSequenceState<Attachment, KvState>> {
        if source.core().position() != expected_position {
            return Err(Error::Execution {
                message: format!(
                    "exact decoder prefix fork expected position {expected_position}, source is at {}",
                    source.core().position()
                ),
            });
        }
        source.logical_fork()
    }
    fn reset(&mut self, state: &mut DecoderSequenceState<Attachment, KvState>) -> Result<()> {
        state.core_mut().reset();
        Ok(())
    }
    fn try_release(
        &mut self,
        state: DecoderSequenceState<Attachment, KvState>,
    ) -> std::result::Result<(), SequenceStateReleaseError<DecoderSequenceState<Attachment, KvState>>>
    {
        match state.try_release_attachment() {
            Ok(_kv_state) => Ok(()),
            Err(error) => {
                let (source, state) = error.into_parts();
                Err(SequenceStateReleaseError::new(source, state))
            }
        }
    }
}
impl DecoderSequenceAttachment for () {
    type Release = ();
    fn preflight_release(&self) -> Result<Self::Release> {
        Ok(())
    }
    fn release(self, _release: Self::Release) {}
}
/// Failed sequence-attachment release retaining the exact original state.
#[derive(Debug)]
#[must_use = "a failed decoder sequence release retains state custody"]
pub struct DecoderSequenceReleaseError<Attachment, KvState> {
    source: Error,
    state: DecoderSequenceState<Attachment, KvState>,
}
impl<Attachment, KvState> DecoderSequenceReleaseError<Attachment, KvState> {
    pub const fn source_error(&self) -> &Error {
        &self.source
    }
    pub const fn state(&self) -> &DecoderSequenceState<Attachment, KvState> {
        &self.state
    }
    pub fn into_parts(self) -> (Error, DecoderSequenceState<Attachment, KvState>) {
        (self.source, self.state)
    }
    pub fn into_source(self) -> Error {
        self.source
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_fork_and_working_copy_have_distinct_identity_semantics() {
        let state = DecoderSequenceState::with_position(7, vec![1], vec![2]);
        let working = state.transaction_working_copy().unwrap();
        let fork = state.logical_fork().unwrap();
        assert_eq!(working.topology_id(), state.topology_id());
        assert_eq!(working.core(), state.core());
        assert_ne!(fork.topology_id(), state.topology_id());
        assert_eq!(fork.core().position(), state.core().position());
        assert_ne!(fork.core().generation(), state.core().generation());
    }
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Attachment {
        allow_release: bool,
    }
    impl DecoderSequenceAttachment for Attachment {
        type Release = ();
        fn preflight_release(&self) -> Result<Self::Release> {
            if self.allow_release {
                Ok(())
            } else {
                Err(Error::Execution {
                    message: "release blocked".into(),
                })
            }
        }
        fn release(self, _release: Self::Release) {}
    }
    #[test]
    fn release_preflight_failure_retains_exact_state() {
        let state = DecoderSequenceState::new(
            Attachment {
                allow_release: false,
            },
            vec![3],
        );
        let topology = state.topology_id();
        let error = state.try_release_attachment().unwrap_err();
        assert_eq!(error.state().topology_id(), topology);
        assert_eq!(error.state().kv_state(), &[3]);
        assert!(error.source_error().to_string().contains("release blocked"));
    }
}
