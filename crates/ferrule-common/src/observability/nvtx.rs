//! Private, host-only NVTX adapter. Process ranges tolerate reentrant and
//! concurrent enters of a span, and resumption on a different thread. Each enter
//! gets its own range; a suspended span is not mislabeled as executing work.
//! Only fixed operation names and unsigned numeric identities/counts are sent.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt::Write;
use std::sync::{Arc, OnceLock};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Metadata, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

pub(super) fn runtime_enabled() -> bool {
    enabled_value(std::env::var("FERRULE_PROFILE_NVTX").ok().as_deref())
}

fn enabled_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(super) fn allows(metadata: &Metadata<'_>) -> bool {
    metadata.is_span()
        && metadata.target() == "ferrule_profile"
        && matches!(
            metadata.name(),
            "read_subset" | "layer.expert" | "layer.expert.upload"
        )
}

const FIELDS: [&str; 7] = [
    "operation",
    "source_generation",
    "requested_bytes",
    "image_generation",
    "layer",
    "expert",
    "rows",
];

#[derive(Default)]
struct NumericFields([Option<u64>; FIELDS.len()]);

impl Visit for NumericFields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if let Some(index) = FIELDS.iter().position(|name| *name == field.name()) {
            self.0[index] = Some(value);
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if let Ok(value) = u64::try_from(value) {
            self.record_u64(field, value);
        }
    }

    // Never format arbitrary Debug values, strings, errors, paths or payloads.
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

struct ProfileSpan {
    key: Arc<()>,
    name: &'static str,
    fields: NumericFields,
}

impl ProfileSpan {
    fn label(&self) -> CString {
        // Fixed names plus at most seven 20-digit u64 values: bounded < 512 bytes.
        let mut label = String::with_capacity(512);
        label.push_str(self.name);
        for (name, value) in FIELDS.iter().zip(self.fields.0) {
            if let Some(value) = value {
                write!(label, " {name}={value}").expect("write to String");
            }
        }
        CString::new(label).expect("fixed names and numeric fields have no NUL")
    }
}

trait RangeBackend: Send + Sync + 'static {
    fn start(&self, label: &CStr) -> u64;
    fn end(&self, range: u64);
}

struct ActiveRange {
    key: Arc<()>,
    range: u64,
}

thread_local! {
    // No process-wide range mutex or CUDA synchronization. Identity, rather
    // than span ID alone, prevents collisions between scoped dispatchers.
    static ACTIVE: RefCell<Vec<ActiveRange>> = const { RefCell::new(Vec::new()) };
}

pub(super) struct NvtxLayer<B = HostNvtx> {
    backend: B,
    enabled: bool,
}

impl NvtxLayer<HostNvtx> {
    pub(super) fn new() -> Self {
        Self {
            backend: HostNvtx::new(),
            enabled: true,
        }
    }
}

impl<S, B> Layer<S> for NvtxLayer<B>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    B: RangeBackend,
{
    fn on_new_span(&self, attributes: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        if !self.enabled || !allows(attributes.metadata()) {
            return;
        }
        if let Some(span) = context.span(id) {
            let mut fields = NumericFields::default();
            attributes.record(&mut fields);
            span.extensions_mut().insert(ProfileSpan {
                key: Arc::new(()),
                name: attributes.metadata().name(),
                fields,
            });
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, context: Context<'_, S>) {
        if let Some(span) = context.span(id)
            && let Some(profile) = span.extensions_mut().get_mut::<ProfileSpan>()
        {
            values.record(&mut profile.fields);
        }
    }

    fn on_enter(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else { return };
        let (key, label) = {
            let extensions = span.extensions();
            let Some(profile) = extensions.get::<ProfileSpan>() else {
                return;
            };
            (Arc::clone(&profile.key), profile.label())
        };
        let range = self.backend.start(&label);
        ACTIVE.with(|active| active.borrow_mut().push(ActiveRange { key, range }));
    }

    fn on_exit(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else { return };
        let key = {
            let extensions = span.extensions();
            let Some(profile) = extensions.get::<ProfileSpan>() else {
                return;
            };
            Arc::clone(&profile.key)
        };
        let range = ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            let index = active
                .iter()
                .rposition(|range| Arc::ptr_eq(&range.key, &key))?;
            Some(active.remove(index).range)
        });
        if let Some(range) = range {
            self.backend.end(range);
        }
    }
}

#[allow(unsafe_code)]
unsafe extern "C" {
    fn ferrule_nvtx_domain_create() -> *mut c_void;
    fn ferrule_nvtx_range_start(domain: *mut c_void, label: *const c_char) -> u64;
    fn ferrule_nvtx_range_end(domain: *mut c_void, range: u64);
}

// NVTX domains are thread-safe, process-lifetime tool handles, not CUDA state.
// Keep the domain alive so ranges from different subscriber lifetimes can end.
static DOMAIN: OnceLock<usize> = OnceLock::new();

pub(super) struct HostNvtx {
    domain: usize,
}

impl HostNvtx {
    #[allow(unsafe_code)]
    fn new() -> Self {
        // SAFETY: the shim returns an opaque, process-lifetime NVTX domain.
        let domain = *DOMAIN.get_or_init(|| unsafe { ferrule_nvtx_domain_create() as usize });
        Self { domain }
    }
}

impl RangeBackend for HostNvtx {
    #[allow(unsafe_code)]
    fn start(&self, label: &CStr) -> u64 {
        // SAFETY: the domain is live and the C string remains valid for the call.
        unsafe { ferrule_nvtx_range_start(self.domain as *mut c_void, label.as_ptr()) }
    }

    #[allow(unsafe_code)]
    fn end(&self, range: u64) {
        // SAFETY: each ID came from start on this domain; zero is a valid no-op
        // when no tool is injected. NVTX process ranges may end across threads.
        unsafe { ferrule_nvtx_range_end(self.domain as *mut c_void, range) }
    }
}

#[cfg(test)]
#[path = "nvtx_tests.rs"]
mod tests;
