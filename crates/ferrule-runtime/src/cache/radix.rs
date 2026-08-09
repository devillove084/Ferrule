//! Model-independent compressed token-prefix cache.
//!
//! The index stores token keys and caller-owned payloads only. It has no model
//! types, backend residency state, block tables, or CUDA physical slots. A
//! typical runtime payload is an opaque `KvPrefixSnapshot` identity whose page
//! lifetime remains owned by `KvPageManager`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::AtomicU64;

use super::take_permanent_identity;

/// Token identity used by the radix index.
pub type PrefixToken = u32;

/// Exact namespace for reusable KV state.
///
/// The values are caller-assigned stable identities or fingerprints. A token
/// sequence is reusable only when the complete model placement, execution plan,
/// and KV owner all match exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrefixCacheNamespace {
    model: u64,
    backend: u64,
    device: u64,
    plan: u64,
    layout: u64,
}

impl PrefixCacheNamespace {
    /// Construct a model-neutral namespace without a placement backend/device.
    pub const fn new(model: u64, plan: u64, layout: u64) -> Self {
        Self {
            model,
            backend: 0,
            device: 0,
            plan,
            layout,
        }
    }

    /// Construct an exact runtime namespace from a complete model placement.
    pub const fn for_placement(
        model: u64,
        backend: u64,
        device: u64,
        plan: u64,
        layout: u64,
    ) -> Self {
        Self {
            model,
            backend,
            device,
            plan,
            layout,
        }
    }

    pub const fn model(self) -> u64 {
        self.model
    }

    pub const fn backend(self) -> u64 {
        self.backend
    }

    pub const fn device(self) -> u64 {
        self.device
    }

    pub const fn plan(self) -> u64 {
        self.plan
    }

    pub const fn layout(self) -> u64 {
        self.layout
    }
}

/// Upper bound applied before longest-prefix lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixLookupLimit {
    /// Every prompt token may be reused.
    AllPromptTokens,
    /// Reuse at most `prompt.len() - 1` tokens.
    ///
    /// This mode leaves the final prompt token for forward execution so a
    /// next-token logits row can be produced even on an otherwise full hit.
    BeforeLastPromptToken,
}

impl PrefixLookupLimit {
    pub const fn reusable_tokens(self, prompt_len: usize) -> usize {
        match self {
            Self::AllPromptTokens => prompt_len,
            Self::BeforeLastPromptToken => prompt_len.saturating_sub(1),
        }
    }
}

/// Stable identity of one resident radix entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrefixCacheEntryId(NonZeroU64);

impl PrefixCacheEntryId {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// A pinned longest-prefix hit.
///
/// This token is intentionally non-cloneable. It must be returned with
/// [`RadixPrefixCache::unpin`]; dropping it leaves the entry pinned rather than
/// performing hidden cache mutation.
#[must_use = "prefix cache leases must be explicitly unpinned"]
#[derive(Debug)]
pub struct PrefixCacheLease {
    cache_id: u64,
    entry_id: PrefixCacheEntryId,
    matched_tokens: usize,
    lru_generation: u64,
}

impl PrefixCacheLease {
    pub const fn entry_id(&self) -> PrefixCacheEntryId {
        self.entry_id
    }

    pub const fn matched_tokens(&self) -> usize {
        self.matched_tokens
    }

    /// LRU generation assigned by the lookup that created this lease.
    pub const fn lru_generation(&self) -> u64 {
        self.lru_generation
    }
}

/// Cache operation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixCacheError {
    EmptyTokenSequence,
    ZeroCharge,
    EntryNotFound,
    EntryIdExhausted,
    EntryPinned {
        entry_id: PrefixCacheEntryId,
        leases: u32,
    },
    InsufficientEvictableCapacity {
        capacity: usize,
        required: usize,
        evictable: usize,
    },
    CapacityOverflow,
    LeasePinOverflow {
        entry_id: PrefixCacheEntryId,
    },
    ForeignLease,
    StaleLease {
        entry_id: PrefixCacheEntryId,
    },
    LeaseUnderflow {
        entry_id: PrefixCacheEntryId,
    },
}

impl fmt::Display for PrefixCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTokenSequence => write!(f, "prefix cache keys cannot be empty"),
            Self::ZeroCharge => write!(f, "prefix cache entries must have non-zero charge"),
            Self::EntryNotFound => write!(f, "prefix cache entry does not exist"),
            Self::EntryIdExhausted => write!(f, "prefix cache entry ID space exhausted"),
            Self::EntryPinned { entry_id, leases } => write!(
                f,
                "prefix cache entry {} is pinned by {leases} lease(s)",
                entry_id.get()
            ),
            Self::InsufficientEvictableCapacity {
                capacity,
                required,
                evictable,
            } => write!(
                f,
                "prefix cache capacity {capacity} cannot admit usage {required}; only {evictable} charge is evictable"
            ),
            Self::CapacityOverflow => write!(f, "prefix cache capacity accounting overflow"),
            Self::LeasePinOverflow { entry_id } => write!(
                f,
                "prefix cache entry {} lease count overflow",
                entry_id.get()
            ),
            Self::ForeignLease => write!(f, "prefix cache lease belongs to another cache"),
            Self::StaleLease { entry_id } => {
                write!(
                    f,
                    "prefix cache lease for entry {} is stale",
                    entry_id.get()
                )
            }
            Self::LeaseUnderflow { entry_id } => write!(
                f,
                "prefix cache entry {} has no lease to unpin",
                entry_id.get()
            ),
        }
    }
}

impl std::error::Error for PrefixCacheError {}

/// Failed insertion retaining ownership of the proposed payload.
#[derive(Debug)]
pub struct PrefixCacheInsertError<P> {
    error: PrefixCacheError,
    payload: P,
}

impl<P> PrefixCacheInsertError<P> {
    pub const fn error(&self) -> &PrefixCacheError {
        &self.error
    }

    pub fn into_parts(self) -> (PrefixCacheError, P) {
        (self.error, self.payload)
    }
}

impl<P> fmt::Display for PrefixCacheInsertError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl<P: fmt::Debug> std::error::Error for PrefixCacheInsertError<P> {}

/// Failed unpin retaining the linear lease token.
#[derive(Debug)]
pub struct PrefixCacheUnpinError {
    error: PrefixCacheError,
    lease: PrefixCacheLease,
}

impl PrefixCacheUnpinError {
    pub const fn error(&self) -> &PrefixCacheError {
        &self.error
    }

    pub fn into_parts(self) -> (PrefixCacheError, PrefixCacheLease) {
        (self.error, self.lease)
    }
}

impl fmt::Display for PrefixCacheUnpinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for PrefixCacheUnpinError {}

/// Payload and key returned when an entry leaves the cache.
#[must_use = "removed prefix payloads may require explicit external release"]
#[derive(Debug)]
pub struct RemovedPrefixEntry<P> {
    namespace: PrefixCacheNamespace,
    tokens: Vec<PrefixToken>,
    payload: P,
    charge: usize,
    last_used_generation: u64,
}

impl<P> RemovedPrefixEntry<P> {
    pub const fn namespace(&self) -> PrefixCacheNamespace {
        self.namespace
    }

    pub fn tokens(&self) -> &[PrefixToken] {
        &self.tokens
    }

    pub const fn payload(&self) -> &P {
        &self.payload
    }

    pub const fn charge(&self) -> usize {
        self.charge
    }

    pub const fn last_used_generation(&self) -> u64 {
        self.last_used_generation
    }

    pub fn into_payload(self) -> P {
        self.payload
    }

    pub fn into_parts(self) -> (PrefixCacheNamespace, Vec<PrefixToken>, P, usize) {
        (self.namespace, self.tokens, self.payload, self.charge)
    }
}

/// Successful insert or replace, including every payload displaced by it.
#[must_use = "replaced and evicted prefix payloads must be explicitly handled"]
#[derive(Debug)]
pub struct PrefixCacheInsertOutcome<P> {
    entry_id: PrefixCacheEntryId,
    replaced: Option<RemovedPrefixEntry<P>>,
    evicted: Vec<RemovedPrefixEntry<P>>,
}

impl<P> PrefixCacheInsertOutcome<P> {
    pub const fn entry_id(&self) -> PrefixCacheEntryId {
        self.entry_id
    }

    pub const fn replaced(&self) -> Option<&RemovedPrefixEntry<P>> {
        self.replaced.as_ref()
    }

    pub fn evicted(&self) -> &[RemovedPrefixEntry<P>] {
        &self.evicted
    }

    pub fn into_parts(
        self,
    ) -> (
        PrefixCacheEntryId,
        Option<RemovedPrefixEntry<P>>,
        Vec<RemovedPrefixEntry<P>>,
    ) {
        (self.entry_id, self.replaced, self.evicted)
    }
}

/// Read-only metadata for the current least-recently-used evictable entry.
#[derive(Debug, Clone, Copy)]
pub struct PrefixCacheEvictionCandidate<'a> {
    entry_id: PrefixCacheEntryId,
    namespace: PrefixCacheNamespace,
    tokens: &'a [PrefixToken],
    charge: usize,
    last_used_generation: u64,
}

impl<'a> PrefixCacheEvictionCandidate<'a> {
    pub const fn entry_id(self) -> PrefixCacheEntryId {
        self.entry_id
    }

    pub const fn namespace(self) -> PrefixCacheNamespace {
        self.namespace
    }

    pub const fn tokens(self) -> &'a [PrefixToken] {
        self.tokens
    }

    pub const fn charge(self) -> usize {
        self.charge
    }

    pub const fn last_used_generation(self) -> u64 {
        self.last_used_generation
    }
}

#[derive(Debug, Default)]
struct RadixNode {
    edge: Vec<PrefixToken>,
    entry: Option<PrefixCacheEntryId>,
    children: BTreeMap<PrefixToken, RadixNode>,
}

#[derive(Debug)]
struct PrefixEntry<P> {
    namespace: PrefixCacheNamespace,
    tokens: Vec<PrefixToken>,
    payload: P,
    charge: usize,
    pin_count: u32,
    last_used_generation: u64,
}

static NEXT_PREFIX_CACHE_ID: AtomicU64 = AtomicU64::new(1);

fn take_prefix_cache_identity(counter: &AtomicU64) -> u64 {
    take_permanent_identity(counter, "prefix-cache identity space exhausted")
}

/// Compressed token-radix cache with weighted capacity and explicit leases.
///
/// Call [`Self::drain`] before shutdown when payloads require explicit external
/// release; the cache deliberately has no resource-releasing `Drop` behavior.
/// Capacity is measured in caller-defined non-zero charge units. Insertions
/// enforce the configured capacity by returning unpinned LRU payloads to the
/// caller. If pinned entries prevent enough eviction, insertion fails without
/// changing entries, usage, pins, or payload ownership.
pub struct RadixPrefixCache<P> {
    cache_id: u64,
    capacity: usize,
    used_capacity: usize,
    lru_generation: u64,
    next_entry_id: NonZeroU64,
    namespaces: HashMap<PrefixCacheNamespace, RadixNode>,
    entries: HashMap<PrefixCacheEntryId, PrefixEntry<P>>,
}

impl<P> fmt::Debug for RadixPrefixCache<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RadixPrefixCache")
            .field("capacity", &self.capacity)
            .field("used_capacity", &self.used_capacity)
            .field("lru_generation", &self.lru_generation)
            .field("namespaces", &self.namespaces.len())
            .field("entries", &self.entries.len())
            .finish_non_exhaustive()
    }
}

impl<P> RadixPrefixCache<P> {
    /// Create a cache with a strict weighted capacity. Zero disables insertion.
    pub fn new(capacity: usize) -> Self {
        let cache_id = take_prefix_cache_identity(&NEXT_PREFIX_CACHE_ID);
        Self {
            cache_id,
            capacity,
            used_capacity: 0,
            lru_generation: 0,
            next_entry_id: NonZeroU64::new(1).expect("one is non-zero"),
            namespaces: HashMap::new(),
            entries: HashMap::new(),
        }
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn used_capacity(&self) -> usize {
        self.used_capacity
    }

    pub const fn lru_generation(&self) -> u64 {
        self.lru_generation
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn namespace_count(&self) -> usize {
        self.namespaces.len()
    }

    /// Insert an exact token frontier, replacing an unpinned exact entry.
    ///
    /// `charge` is an arbitrary caller-defined capacity unit, commonly retained
    /// pages or bytes. Every replaced or LRU-evicted payload is returned so its
    /// external resources can be explicitly released.
    pub fn insert(
        &mut self,
        namespace: PrefixCacheNamespace,
        tokens: &[PrefixToken],
        payload: P,
        charge: usize,
    ) -> Result<PrefixCacheInsertOutcome<P>, PrefixCacheInsertError<P>> {
        let prepared = self.prepare_insert(namespace, tokens, charge);
        let (existing, evictions, new_identity) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Err(PrefixCacheInsertError { error, payload }),
        };

        let generation = self.next_lru_generation();
        let mut evicted = Vec::with_capacity(evictions.len());
        for entry_id in evictions {
            evicted.push(self.remove_entry_unchecked(entry_id));
        }

        if let Some(entry_id) = existing {
            let entry = self
                .entries
                .get_mut(&entry_id)
                .expect("prepared replacement entry must remain resident");
            let old_charge = entry.charge;
            let old_generation = entry.last_used_generation;
            let old_payload = std::mem::replace(&mut entry.payload, payload);
            entry.charge = charge;
            entry.last_used_generation = generation;
            self.used_capacity = self
                .used_capacity
                .checked_sub(old_charge)
                .and_then(|usage| usage.checked_add(charge))
                .expect("prepared replacement capacity must remain valid");
            let replaced = RemovedPrefixEntry {
                namespace,
                tokens: tokens.to_vec(),
                payload: old_payload,
                charge: old_charge,
                last_used_generation: old_generation,
            };
            return Ok(PrefixCacheInsertOutcome {
                entry_id,
                replaced: Some(replaced),
                evicted,
            });
        }

        let (entry_id, next_entry_id) = new_identity.expect("new entry identity was prepared");
        self.next_entry_id = next_entry_id;
        let root = self.namespaces.entry(namespace).or_default();
        insert_radix_entry(root, tokens, entry_id);
        assert!(
            self.entries
                .insert(
                    entry_id,
                    PrefixEntry {
                        namespace,
                        tokens: tokens.to_vec(),
                        payload,
                        charge,
                        pin_count: 0,
                        last_used_generation: generation,
                    },
                )
                .is_none(),
            "fresh prefix entry ID must be vacant"
        );
        self.used_capacity = self
            .used_capacity
            .checked_add(charge)
            .expect("prepared insertion capacity must remain valid");
        Ok(PrefixCacheInsertOutcome {
            entry_id,
            replaced: None,
            evicted,
        })
    }

    /// Replace an existing exact entry, returning its old payload and any LRU evictions.
    ///
    /// Unlike [`Self::insert`], this operation fails without mutation when the
    /// exact key is absent. On every failure the proposed payload is returned.
    pub fn replace(
        &mut self,
        namespace: PrefixCacheNamespace,
        tokens: &[PrefixToken],
        payload: P,
        charge: usize,
    ) -> Result<PrefixCacheInsertOutcome<P>, PrefixCacheInsertError<P>> {
        if !self.contains_exact(namespace, tokens) {
            return Err(PrefixCacheInsertError {
                error: PrefixCacheError::EntryNotFound,
                payload,
            });
        }
        let outcome = self.insert(namespace, tokens, payload, charge)?;
        debug_assert!(outcome.replaced.is_some());
        Ok(outcome)
    }

    /// Pin and return the longest exact cached prefix under the selected limit.
    pub fn lookup_longest_prefix(
        &mut self,
        namespace: PrefixCacheNamespace,
        prompt: &[PrefixToken],
        limit: PrefixLookupLimit,
    ) -> Result<Option<PrefixCacheLease>, PrefixCacheError> {
        let reusable = limit.reusable_tokens(prompt.len());
        if reusable == 0 {
            return Ok(None);
        }
        let Some((entry_id, matched_tokens)) = self
            .namespaces
            .get(&namespace)
            .and_then(|root| longest_radix_prefix(root, &prompt[..reusable]))
        else {
            return Ok(None);
        };
        let entry = self
            .entries
            .get(&entry_id)
            .expect("radix entry must have resident metadata");
        if entry.pin_count == u32::MAX {
            return Err(PrefixCacheError::LeasePinOverflow { entry_id });
        }
        debug_assert_eq!(entry.tokens.len(), matched_tokens);
        let generation = self.next_lru_generation();
        let entry = self
            .entries
            .get_mut(&entry_id)
            .expect("looked-up entry must remain resident");
        entry.pin_count += 1;
        entry.last_used_generation = generation;
        Ok(Some(PrefixCacheLease {
            cache_id: self.cache_id,
            entry_id,
            matched_tokens,
            lru_generation: generation,
        }))
    }

    /// Borrow the generic payload protected by a live lease.
    pub fn payload(&self, lease: &PrefixCacheLease) -> Result<&P, PrefixCacheError> {
        if lease.cache_id != self.cache_id {
            return Err(PrefixCacheError::ForeignLease);
        }
        let entry = self
            .entries
            .get(&lease.entry_id)
            .ok_or(PrefixCacheError::StaleLease {
                entry_id: lease.entry_id,
            })?;
        if entry.pin_count == 0 {
            return Err(PrefixCacheError::LeaseUnderflow {
                entry_id: lease.entry_id,
            });
        }
        Ok(&entry.payload)
    }

    /// Explicitly release one entry pin.
    pub fn unpin(&mut self, lease: PrefixCacheLease) -> Result<(), PrefixCacheUnpinError> {
        let error = if lease.cache_id != self.cache_id {
            Some(PrefixCacheError::ForeignLease)
        } else if let Some(entry) = self.entries.get_mut(&lease.entry_id) {
            if entry.pin_count == 0 {
                Some(PrefixCacheError::LeaseUnderflow {
                    entry_id: lease.entry_id,
                })
            } else {
                entry.pin_count -= 1;
                None
            }
        } else {
            Some(PrefixCacheError::StaleLease {
                entry_id: lease.entry_id,
            })
        };
        match error {
            Some(error) => Err(PrefixCacheUnpinError { error, lease }),
            None => Ok(()),
        }
    }

    pub fn entry_pin_count(&self, entry_id: PrefixCacheEntryId) -> Option<u32> {
        self.entries.get(&entry_id).map(|entry| entry.pin_count)
    }

    pub fn contains_exact(&self, namespace: PrefixCacheNamespace, tokens: &[PrefixToken]) -> bool {
        self.namespaces
            .get(&namespace)
            .and_then(|root| exact_radix_entry(root, tokens))
            .is_some()
    }

    /// Remove an exact unpinned entry and return its payload.
    pub fn remove(
        &mut self,
        namespace: PrefixCacheNamespace,
        tokens: &[PrefixToken],
    ) -> Result<Option<RemovedPrefixEntry<P>>, PrefixCacheError> {
        let Some(entry_id) = self
            .namespaces
            .get(&namespace)
            .and_then(|root| exact_radix_entry(root, tokens))
        else {
            return Ok(None);
        };
        let pin_count = self
            .entries
            .get(&entry_id)
            .expect("radix entry must have resident metadata")
            .pin_count;
        if pin_count != 0 {
            return Err(PrefixCacheError::EntryPinned {
                entry_id,
                leases: pin_count,
            });
        }
        Ok(Some(self.remove_entry_unchecked(entry_id)))
    }

    /// Return the globally least-recently-used unpinned candidate.
    pub fn lru_evictable(&self) -> Option<PrefixCacheEvictionCandidate<'_>> {
        let (entry_id, entry) = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.pin_count == 0)
            .min_by_key(|(entry_id, entry)| (entry.last_used_generation, **entry_id))?;
        Some(PrefixCacheEvictionCandidate {
            entry_id: *entry_id,
            namespace: entry.namespace,
            tokens: &entry.tokens,
            charge: entry.charge,
            last_used_generation: entry.last_used_generation,
        })
    }

    /// Evict and return the globally least-recently-used unpinned entry.
    pub fn evict_lru(&mut self) -> Option<RemovedPrefixEntry<P>> {
        let entry_id = self.lru_evictable()?.entry_id;
        Some(self.remove_entry_unchecked(entry_id))
    }

    /// Remove every entry and return all payloads in deterministic LRU order.
    ///
    /// This is the explicit shutdown path for payloads that own external
    /// resources such as KV prefix snapshots. It fails without mutation while
    /// any entry is leased.
    pub fn drain(&mut self) -> Result<Vec<RemovedPrefixEntry<P>>, PrefixCacheError> {
        if let Some((entry_id, entry)) = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.pin_count != 0)
            .min_by_key(|(entry_id, _)| **entry_id)
        {
            return Err(PrefixCacheError::EntryPinned {
                entry_id: *entry_id,
                leases: entry.pin_count,
            });
        }
        let mut entry_ids = self
            .entries
            .iter()
            .map(|(entry_id, entry)| (*entry_id, entry.last_used_generation))
            .collect::<Vec<_>>();
        entry_ids.sort_unstable_by_key(|(entry_id, generation)| (*generation, *entry_id));
        Ok(entry_ids
            .into_iter()
            .map(|(entry_id, _)| self.remove_entry_unchecked(entry_id))
            .collect())
    }

    /// Atomically change capacity, evicting unpinned LRU entries if needed.
    ///
    /// If pins prevent enough eviction, neither capacity nor cache contents are
    /// changed.
    pub fn set_capacity(
        &mut self,
        capacity: usize,
    ) -> Result<Vec<RemovedPrefixEntry<P>>, PrefixCacheError> {
        let evictions = self.plan_evictions(self.used_capacity, capacity, None)?;
        let mut removed = Vec::with_capacity(evictions.len());
        for entry_id in evictions {
            removed.push(self.remove_entry_unchecked(entry_id));
        }
        self.capacity = capacity;
        Ok(removed)
    }

    fn prepare_insert(
        &self,
        namespace: PrefixCacheNamespace,
        tokens: &[PrefixToken],
        charge: usize,
    ) -> Result<PreparedInsert, PrefixCacheError> {
        if tokens.is_empty() {
            return Err(PrefixCacheError::EmptyTokenSequence);
        }
        if charge == 0 {
            return Err(PrefixCacheError::ZeroCharge);
        }
        let existing = self
            .namespaces
            .get(&namespace)
            .and_then(|root| exact_radix_entry(root, tokens));
        let old_charge = if let Some(entry_id) = existing {
            let entry = self
                .entries
                .get(&entry_id)
                .expect("radix entry must have resident metadata");
            if entry.pin_count != 0 {
                return Err(PrefixCacheError::EntryPinned {
                    entry_id,
                    leases: entry.pin_count,
                });
            }
            entry.charge
        } else {
            0
        };
        let prospective = self
            .used_capacity
            .checked_sub(old_charge)
            .and_then(|usage| usage.checked_add(charge))
            .ok_or(PrefixCacheError::CapacityOverflow)?;
        let evictions = self.plan_evictions(prospective, self.capacity, existing)?;
        let identity = if existing.is_none() {
            Some(self.prepare_entry_identity()?)
        } else {
            None
        };
        Ok((existing, evictions, identity))
    }

    fn prepare_entry_identity(&self) -> Result<(PrefixCacheEntryId, NonZeroU64), PrefixCacheError> {
        let entry_id = PrefixCacheEntryId(self.next_entry_id);
        let next = self
            .next_entry_id
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or(PrefixCacheError::EntryIdExhausted)?;
        Ok((entry_id, next))
    }

    fn plan_evictions(
        &self,
        prospective_usage: usize,
        capacity: usize,
        excluded: Option<PrefixCacheEntryId>,
    ) -> Result<Vec<PrefixCacheEntryId>, PrefixCacheError> {
        let required_free = prospective_usage.saturating_sub(capacity);
        if required_free == 0 {
            return Ok(Vec::new());
        }
        let mut candidates = self
            .entries
            .iter()
            .filter(|(entry_id, entry)| entry.pin_count == 0 && Some(**entry_id) != excluded)
            .map(|(entry_id, entry)| (*entry_id, entry.last_used_generation, entry.charge))
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(entry_id, generation, _)| (*generation, *entry_id));
        let evictable = candidates
            .iter()
            .try_fold(0usize, |total, (_, _, charge)| {
                total
                    .checked_add(*charge)
                    .ok_or(PrefixCacheError::CapacityOverflow)
            })?;
        if evictable < required_free {
            return Err(PrefixCacheError::InsufficientEvictableCapacity {
                capacity,
                required: prospective_usage,
                evictable,
            });
        }
        let mut selected = Vec::new();
        let mut selected_charge = 0usize;
        for (entry_id, _, charge) in candidates {
            selected.push(entry_id);
            selected_charge = selected_charge
                .checked_add(charge)
                .ok_or(PrefixCacheError::CapacityOverflow)?;
            if selected_charge >= required_free {
                break;
            }
        }
        Ok(selected)
    }

    fn next_lru_generation(&mut self) -> u64 {
        if self.lru_generation == u64::MAX {
            let mut ordered = self
                .entries
                .iter()
                .map(|(entry_id, entry)| (*entry_id, entry.last_used_generation))
                .collect::<Vec<_>>();
            ordered.sort_unstable_by_key(|(entry_id, generation)| (*generation, *entry_id));
            for (index, (entry_id, _)) in ordered.into_iter().enumerate() {
                self.entries
                    .get_mut(&entry_id)
                    .expect("generation entry must remain resident")
                    .last_used_generation = index as u64 + 1;
            }
            self.lru_generation = self.entries.len() as u64;
        }
        self.lru_generation += 1;
        self.lru_generation
    }

    fn remove_entry_unchecked(&mut self, entry_id: PrefixCacheEntryId) -> RemovedPrefixEntry<P> {
        let entry = self
            .entries
            .remove(&entry_id)
            .expect("selected prefix entry must remain resident");
        assert_eq!(entry.pin_count, 0, "pinned prefix entry cannot be removed");
        let namespace_empty = {
            let root = self
                .namespaces
                .get_mut(&entry.namespace)
                .expect("prefix entry namespace must remain resident");
            let removed = clear_radix_entry(root, &entry.tokens);
            assert_eq!(removed, Some(entry_id), "radix metadata must match entry");
            root.entry.is_none() && root.children.is_empty()
        };
        if namespace_empty {
            self.namespaces.remove(&entry.namespace);
        }
        self.used_capacity = self
            .used_capacity
            .checked_sub(entry.charge)
            .expect("resident entry charge must be accounted");
        RemovedPrefixEntry {
            namespace: entry.namespace,
            tokens: entry.tokens,
            payload: entry.payload,
            charge: entry.charge,
            last_used_generation: entry.last_used_generation,
        }
    }
}

type PreparedInsert = (
    Option<PrefixCacheEntryId>,
    Vec<PrefixCacheEntryId>,
    Option<(PrefixCacheEntryId, NonZeroU64)>,
);

fn common_prefix_len(left: &[PrefixToken], right: &[PrefixToken]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn exact_radix_entry(root: &RadixNode, tokens: &[PrefixToken]) -> Option<PrefixCacheEntryId> {
    let mut node = root;
    let mut suffix = tokens;
    while let Some(first) = suffix.first() {
        let child = node.children.get(first)?;
        if !suffix.starts_with(&child.edge) {
            return None;
        }
        suffix = &suffix[child.edge.len()..];
        node = child;
    }
    node.entry
}

fn longest_radix_prefix(
    root: &RadixNode,
    tokens: &[PrefixToken],
) -> Option<(PrefixCacheEntryId, usize)> {
    let mut node = root;
    let mut consumed = 0usize;
    let mut longest = root.entry.map(|entry_id| (entry_id, 0));
    while consumed < tokens.len() {
        let Some(child) = node.children.get(&tokens[consumed]) else {
            break;
        };
        let remaining = &tokens[consumed..];
        if !remaining.starts_with(&child.edge) {
            break;
        }
        consumed += child.edge.len();
        node = child;
        if let Some(entry_id) = node.entry {
            longest = Some((entry_id, consumed));
        }
    }
    longest
}

fn insert_radix_entry(node: &mut RadixNode, tokens: &[PrefixToken], entry_id: PrefixCacheEntryId) {
    if tokens.is_empty() {
        assert!(node.entry.replace(entry_id).is_none());
        return;
    }
    let first = tokens[0];
    let Some(child) = node.children.get(&first) else {
        node.children.insert(
            first,
            RadixNode {
                edge: tokens.to_vec(),
                entry: Some(entry_id),
                children: BTreeMap::new(),
            },
        );
        return;
    };
    let shared = common_prefix_len(&child.edge, tokens);
    assert!(shared > 0, "child map key must match its first edge token");
    if shared == child.edge.len() {
        let child = node.children.get_mut(&first).expect("child was observed");
        insert_radix_entry(child, &tokens[shared..], entry_id);
        return;
    }

    let mut old_child = node.children.remove(&first).expect("child was observed");
    let old_suffix = old_child.edge.split_off(shared);
    let shared_edge = std::mem::replace(&mut old_child.edge, old_suffix);
    let old_key = old_child.edge[0];
    let mut branch = RadixNode {
        edge: shared_edge,
        entry: None,
        children: BTreeMap::new(),
    };
    branch.children.insert(old_key, old_child);
    if shared == tokens.len() {
        branch.entry = Some(entry_id);
    } else {
        let new_suffix = tokens[shared..].to_vec();
        branch.children.insert(
            new_suffix[0],
            RadixNode {
                edge: new_suffix,
                entry: Some(entry_id),
                children: BTreeMap::new(),
            },
        );
    }
    assert!(node.children.insert(first, branch).is_none());
}

fn clear_radix_entry(node: &mut RadixNode, tokens: &[PrefixToken]) -> Option<PrefixCacheEntryId> {
    if tokens.is_empty() {
        return node.entry.take();
    }
    let first = tokens[0];
    let edge_len = {
        let child = node.children.get(&first)?;
        if !tokens.starts_with(&child.edge) {
            return None;
        }
        child.edge.len()
    };
    let removed = {
        let child = node.children.get_mut(&first).expect("child was observed");
        clear_radix_entry(child, &tokens[edge_len..])
    };
    if removed.is_some() {
        compact_radix_child(node, first);
    }
    removed
}

fn compact_radix_child(parent: &mut RadixNode, key: PrefixToken) {
    let mut child = parent.children.remove(&key).expect("child was observed");
    while child.entry.is_none() && child.children.len() == 1 {
        let (_, grandchild) = child
            .children
            .pop_first()
            .expect("single child must be present");
        child.edge.extend(grandchild.edge);
        child.entry = grandchild.entry;
        child.children = grandchild.children;
    }
    if child.entry.is_some() || !child.children.is_empty() {
        assert_eq!(child.edge.first().copied(), Some(key));
        assert!(parent.children.insert(key, child).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(model: u64, plan: u64, layout: u64) -> PrefixCacheNamespace {
        PrefixCacheNamespace::new(model, plan, layout)
    }

    fn insert_without_displacement<P: fmt::Debug>(
        cache: &mut RadixPrefixCache<P>,
        namespace: PrefixCacheNamespace,
        tokens: &[PrefixToken],
        payload: P,
        charge: usize,
    ) -> PrefixCacheEntryId {
        let outcome = cache.insert(namespace, tokens, payload, charge).unwrap();
        assert!(outcome.replaced().is_none());
        assert!(outcome.evicted().is_empty());
        outcome.entry_id()
    }

    fn lookup_payload<'a>(
        cache: &'a mut RadixPrefixCache<&'static str>,
        namespace: PrefixCacheNamespace,
        prompt: &[PrefixToken],
        limit: PrefixLookupLimit,
    ) -> Option<(usize, &'a str, PrefixCacheLease)> {
        let lease = cache
            .lookup_longest_prefix(namespace, prompt, limit)
            .unwrap()?;
        let matched = lease.matched_tokens();
        let payload = *cache.payload(&lease).unwrap();
        Some((matched, payload, lease))
    }

    #[test]
    fn compressed_radix_finds_longest_exact_prefix_across_splits() {
        let namespace = ns(1, 2, 3);
        let mut cache = RadixPrefixCache::new(16);
        insert_without_displacement(&mut cache, namespace, &[10, 20, 30, 40], "deep", 1);
        insert_without_displacement(&mut cache, namespace, &[10, 20], "short", 1);
        insert_without_displacement(&mut cache, namespace, &[10, 20, 31], "sibling", 1);
        insert_without_displacement(&mut cache, namespace, &[10, 9], "other", 1);

        let (matched, payload, lease) = lookup_payload(
            &mut cache,
            namespace,
            &[10, 20, 30, 40, 50],
            PrefixLookupLimit::AllPromptTokens,
        )
        .unwrap();
        assert_eq!((matched, payload), (4, "deep"));
        cache.unpin(lease).unwrap();

        let (matched, payload, lease) = lookup_payload(
            &mut cache,
            namespace,
            &[10, 20, 99],
            PrefixLookupLimit::AllPromptTokens,
        )
        .unwrap();
        assert_eq!((matched, payload), (2, "short"));
        cache.unpin(lease).unwrap();
        assert!(
            cache
                .lookup_longest_prefix(namespace, &[10, 8], PrefixLookupLimit::AllPromptTokens)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn token_value_collisions_do_not_create_false_prefix_hits() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(8);
        insert_without_displacement(&mut cache, namespace, &[1, 23], "one-twenty-three", 1);
        insert_without_displacement(&mut cache, namespace, &[12, 3], "twelve-three", 1);
        insert_without_displacement(&mut cache, namespace, &[1, 2, 3], "one-two-three", 1);

        for (prompt, expected) in [
            (&[1, 23, 9][..], "one-twenty-three"),
            (&[12, 3, 9][..], "twelve-three"),
            (&[1, 2, 3, 9][..], "one-two-three"),
        ] {
            let (_, payload, lease) = lookup_payload(
                &mut cache,
                namespace,
                prompt,
                PrefixLookupLimit::AllPromptTokens,
            )
            .unwrap();
            assert_eq!(payload, expected);
            cache.unpin(lease).unwrap();
        }
    }

    #[test]
    fn model_plan_and_layout_namespaces_are_all_isolated() {
        let tokens = [4, 5, 6];
        let namespaces = [ns(1, 2, 3), ns(9, 2, 3), ns(1, 9, 3), ns(1, 2, 9)];
        let mut cache = RadixPrefixCache::new(8);
        for (index, namespace) in namespaces.into_iter().enumerate() {
            insert_without_displacement(&mut cache, namespace, &tokens, index, 1);
        }
        for (index, namespace) in namespaces.into_iter().enumerate() {
            let lease = cache
                .lookup_longest_prefix(namespace, &tokens, PrefixLookupLimit::AllPromptTokens)
                .unwrap()
                .unwrap();
            assert_eq!(*cache.payload(&lease).unwrap(), index);
            cache.unpin(lease).unwrap();
        }
        assert_eq!(cache.namespace_count(), 4);
        assert!(
            cache
                .lookup_longest_prefix(ns(8, 8, 8), &tokens, PrefixLookupLimit::AllPromptTokens)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn full_hit_policy_can_reserve_the_last_prompt_token() {
        let namespace = ns(1, 2, 3);
        let mut cache = RadixPrefixCache::new(8);
        insert_without_displacement(&mut cache, namespace, &[1, 2], "two", 1);
        insert_without_displacement(&mut cache, namespace, &[1, 2, 3], "three", 1);

        let (matched, payload, lease) = lookup_payload(
            &mut cache,
            namespace,
            &[1, 2, 3],
            PrefixLookupLimit::AllPromptTokens,
        )
        .unwrap();
        assert_eq!((matched, payload), (3, "three"));
        cache.unpin(lease).unwrap();

        let (matched, payload, lease) = lookup_payload(
            &mut cache,
            namespace,
            &[1, 2, 3],
            PrefixLookupLimit::BeforeLastPromptToken,
        )
        .unwrap();
        assert_eq!((matched, payload), (2, "two"));
        cache.unpin(lease).unwrap();

        let mut single = RadixPrefixCache::new(1);
        insert_without_displacement(&mut single, namespace, &[7], "single", 1);
        assert!(
            single
                .lookup_longest_prefix(namespace, &[7], PrefixLookupLimit::BeforeLastPromptToken)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn leases_pin_entries_and_lru_skips_them() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(2);
        let a = insert_without_displacement(&mut cache, namespace, &[1], "a", 1);
        insert_without_displacement(&mut cache, namespace, &[2], "b", 1);
        let lease_a = cache
            .lookup_longest_prefix(namespace, &[1, 9], PrefixLookupLimit::AllPromptTokens)
            .unwrap()
            .unwrap();
        assert_eq!(cache.entry_pin_count(a), Some(1));

        let inserted = cache.insert(namespace, &[3], "c", 1).unwrap();
        assert_eq!(inserted.evicted().len(), 1);
        assert_eq!(inserted.evicted()[0].payload(), &"b");
        assert!(cache.contains_exact(namespace, &[1]));
        assert!(cache.contains_exact(namespace, &[3]));
        assert!(!cache.contains_exact(namespace, &[2]));

        let lease_c = cache
            .lookup_longest_prefix(namespace, &[3], PrefixLookupLimit::AllPromptTokens)
            .unwrap()
            .unwrap();
        let error = cache.insert(namespace, &[4], "d", 1).unwrap_err();
        let (error, payload) = error.into_parts();
        assert_eq!(payload, "d");
        assert!(matches!(
            error,
            PrefixCacheError::InsufficientEvictableCapacity { .. }
        ));
        assert!(!cache.contains_exact(namespace, &[4]));

        cache.unpin(lease_c).unwrap();
        let inserted = cache.insert(namespace, &[4], "d", 1).unwrap();
        assert_eq!(inserted.evicted()[0].payload(), &"c");
        cache.unpin(lease_a).unwrap();
        assert_eq!(cache.entry_pin_count(a), Some(0));
    }

    #[test]
    fn lookup_updates_lru_generation_and_candidate_order() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(3);
        insert_without_displacement(&mut cache, namespace, &[1], "a", 1);
        insert_without_displacement(&mut cache, namespace, &[2], "b", 1);
        let before = cache.lru_generation();
        let lease = cache
            .lookup_longest_prefix(namespace, &[1], PrefixLookupLimit::AllPromptTokens)
            .unwrap()
            .unwrap();
        assert!(lease.lru_generation() > before);
        cache.unpin(lease).unwrap();
        let candidate = cache.lru_evictable().unwrap();
        assert_eq!(candidate.tokens(), &[2]);
        assert!(candidate.last_used_generation() < cache.lru_generation());
    }

    #[test]
    fn insert_replaces_exact_payload_and_remove_recompresses_tree() {
        let namespace = ns(4, 5, 6);
        let mut cache = RadixPrefixCache::new(8);
        insert_without_displacement(&mut cache, namespace, &[1, 2], "short", 2);
        insert_without_displacement(&mut cache, namespace, &[1, 2, 3], "old", 3);
        insert_without_displacement(&mut cache, namespace, &[1, 2, 4], "sibling", 1);

        let replacement = cache.replace(namespace, &[1, 2, 3], "new", 1).unwrap();
        let replaced = replacement.replaced().unwrap();
        assert_eq!(replaced.payload(), &"old");
        assert_eq!(replaced.charge(), 3);
        assert_eq!(cache.used_capacity(), 4);

        let removed = cache.remove(namespace, &[1, 2]).unwrap().unwrap();
        assert_eq!(removed.payload(), &"short");
        assert!(!cache.contains_exact(namespace, &[1, 2]));
        assert!(cache.contains_exact(namespace, &[1, 2, 3]));
        assert!(cache.contains_exact(namespace, &[1, 2, 4]));

        let (_, payload, lease) = lookup_payload(
            &mut cache,
            namespace,
            &[1, 2, 3, 9],
            PrefixLookupLimit::AllPromptTokens,
        )
        .unwrap();
        assert_eq!(payload, "new");
        cache.unpin(lease).unwrap();
    }

    #[test]
    fn pinned_entries_cannot_be_replaced_or_removed() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(2);
        insert_without_displacement(&mut cache, namespace, &[1], "old", 1);
        let lease = cache
            .lookup_longest_prefix(namespace, &[1], PrefixLookupLimit::AllPromptTokens)
            .unwrap()
            .unwrap();

        let (error, payload) = cache
            .insert(namespace, &[1], "replacement", 1)
            .unwrap_err()
            .into_parts();
        assert_eq!(payload, "replacement");
        assert!(matches!(error, PrefixCacheError::EntryPinned { .. }));
        assert!(matches!(
            cache.remove(namespace, &[1]),
            Err(PrefixCacheError::EntryPinned { .. })
        ));
        assert_eq!(*cache.payload(&lease).unwrap(), "old");
        cache.unpin(lease).unwrap();
    }

    #[test]
    fn weighted_capacity_evicts_enough_lru_entries() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(5);
        insert_without_displacement(&mut cache, namespace, &[1], "a", 2);
        insert_without_displacement(&mut cache, namespace, &[2], "b", 2);
        insert_without_displacement(&mut cache, namespace, &[3], "c", 1);

        let outcome = cache.insert(namespace, &[4], "d", 4).unwrap();
        let evicted = outcome
            .evicted()
            .iter()
            .map(|entry| *entry.payload())
            .collect::<Vec<_>>();
        assert_eq!(evicted, vec!["a", "b"]);
        assert_eq!(cache.used_capacity(), 5);
        assert!(cache.contains_exact(namespace, &[3]));
        assert!(cache.contains_exact(namespace, &[4]));
    }

    #[test]
    fn drain_returns_all_payloads_and_is_atomic_while_pinned() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(3);
        insert_without_displacement(&mut cache, namespace, &[1], "a", 1);
        insert_without_displacement(&mut cache, namespace, &[2], "b", 1);
        let lease = cache
            .lookup_longest_prefix(namespace, &[1], PrefixLookupLimit::AllPromptTokens)
            .unwrap()
            .unwrap();

        assert!(matches!(
            cache.drain(),
            Err(PrefixCacheError::EntryPinned { .. })
        ));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.used_capacity(), 2);

        cache.unpin(lease).unwrap();
        let removed = cache
            .drain()
            .unwrap()
            .into_iter()
            .map(RemovedPrefixEntry::into_payload)
            .collect::<Vec<_>>();
        assert_eq!(removed, vec!["b", "a"]);
        assert!(cache.is_empty());
        assert_eq!(cache.used_capacity(), 0);
        assert_eq!(cache.namespace_count(), 0);
    }

    #[test]
    fn capacity_reduction_is_atomic_when_pins_block_eviction() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(4);
        insert_without_displacement(&mut cache, namespace, &[1], "a", 2);
        insert_without_displacement(&mut cache, namespace, &[2], "b", 2);
        let lease = cache
            .lookup_longest_prefix(namespace, &[1], PrefixLookupLimit::AllPromptTokens)
            .unwrap()
            .unwrap();

        let error = cache.set_capacity(1).unwrap_err();
        assert!(matches!(
            error,
            PrefixCacheError::InsufficientEvictableCapacity { .. }
        ));
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.used_capacity(), 4);
        assert_eq!(cache.len(), 2);

        cache.unpin(lease).unwrap();
        let removed = cache.set_capacity(1).unwrap();
        assert_eq!(removed.len(), 2);
        assert_eq!(cache.capacity(), 1);
        assert_eq!(cache.used_capacity(), 0);
    }

    #[test]
    fn replace_requires_an_existing_exact_entry() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(2);
        insert_without_displacement(&mut cache, namespace, &[1, 2], "existing", 1);

        let (error, payload) = cache
            .replace(namespace, &[1], "missing", 1)
            .unwrap_err()
            .into_parts();
        assert_eq!(error, PrefixCacheError::EntryNotFound);
        assert_eq!(payload, "missing");
        assert_eq!(cache.len(), 1);
        assert!(cache.contains_exact(namespace, &[1, 2]));
    }

    #[test]
    fn invalid_insertions_return_payload_without_mutation() {
        let namespace = ns(1, 1, 1);
        let mut cache = RadixPrefixCache::new(1);
        let (error, payload) = cache
            .insert(namespace, &[], "empty", 1)
            .unwrap_err()
            .into_parts();
        assert_eq!(error, PrefixCacheError::EmptyTokenSequence);
        assert_eq!(payload, "empty");
        let (error, payload) = cache
            .insert(namespace, &[1], "zero", 0)
            .unwrap_err()
            .into_parts();
        assert_eq!(error, PrefixCacheError::ZeroCharge);
        assert_eq!(payload, "zero");
        assert!(cache.is_empty());
    }

    #[test]
    fn foreign_unpin_returns_the_lease_to_its_owner() {
        let namespace = ns(1, 1, 1);
        let mut owner = RadixPrefixCache::new(1);
        let mut other: RadixPrefixCache<&'static str> = RadixPrefixCache::new(1);
        insert_without_displacement(&mut owner, namespace, &[1], "owner", 1);
        let lease = owner
            .lookup_longest_prefix(namespace, &[1], PrefixLookupLimit::AllPromptTokens)
            .unwrap()
            .unwrap();

        let error = other.unpin(lease).unwrap_err();
        let (kind, lease) = error.into_parts();
        assert_eq!(kind, PrefixCacheError::ForeignLease);
        owner.unpin(lease).unwrap();
        assert_eq!(owner.lru_evictable().unwrap().tokens(), &[1]);
    }

    #[test]
    fn prefix_cache_identity_exhaustion_is_permanent() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(take_prefix_cache_identity(&counter), u64::MAX - 1);
        assert_eq!(take_prefix_cache_identity(&counter), u64::MAX);
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);

        for _ in 0..2 {
            assert!(std::panic::catch_unwind(|| take_prefix_cache_identity(&counter)).is_err());
            assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
        }
    }
}
