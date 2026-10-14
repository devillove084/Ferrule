//! Shared short per-device initialization coordination and persistent owner/family charges.
use crate::cuda::{operator_owner::cu, runtime::CudaStream};
use ferrule_common::{Error, Result};
use std::sync::Arc;
fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("CUDA provider admission: {}", message.into()),
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ProviderFamily {
    GemmEx,
    Attention,
}
unsafe extern "C" {
    pub(crate) fn ferrule_bf16_gemmex_process_config_valid() -> i32;
}
/// Provider-family ledger, not a process-wide lifetime circuit breaker. An
/// exact stream owner admits one backing pool; independently budgeted owners
/// may coexist on the same device or other devices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Charge {
    pub(crate) storage_grant: usize,
    pub(crate) external: usize,
}
impl Charge {
    pub(crate) fn total(self) -> Result<usize> {
        self.storage_grant
            .checked_add(self.external)
            .ok_or_else(|| invalid("reservation overflow"))
    }
    pub(crate) fn add(self, other: Self) -> Result<Self> {
        let charge = Self {
            storage_grant: self
                .storage_grant
                .checked_add(other.storage_grant)
                .ok_or_else(|| invalid("storage ledger overflow"))?,
            external: self
                .external
                .checked_add(other.external)
                .ok_or_else(|| invalid("provider ledger overflow"))?,
        };
        charge.total()?;
        Ok(charge)
    }
}
#[derive(Default)]
pub(crate) struct Ledger {
    pub(crate) owners: std::collections::BTreeMap<(usize, ProviderFamily), (usize, Charge)>,
    pub(crate) devices: std::collections::BTreeMap<usize, Charge>,
}
impl Ledger {
    pub(crate) fn reserve(
        &mut self,
        owner: (usize, ProviderFamily),
        device: usize,
        charge: Charge,
    ) -> Result<()> {
        if self.owners.contains_key(&owner) {
            return Err(invalid(
                "this stream owner already has a reserved pool; reuse it or release every prepared/graph lease before preparing another",
            ));
        }
        let total = self
            .devices
            .get(&device)
            .copied()
            .unwrap_or_default()
            .add(charge)?;
        self.owners.insert(owner, (device, charge));
        self.devices.insert(device, total);
        Ok(())
    }
    pub(crate) fn release(&mut self, owner: (usize, ProviderFamily)) {
        let (device, charge) = self
            .owners
            .remove(&owner)
            .expect("owned GemmEx reservation");
        let total = self.devices.get_mut(&device).expect("device reservation");
        total.storage_grant -= charge.storage_grant;
        total.external -= charge.external;
        if *total == Charge::default() {
            self.devices.remove(&device);
        }
    }
}
static LEDGER: std::sync::LazyLock<std::sync::Mutex<Ledger>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Ledger::default()));
static INITIALIZERS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<usize, Arc<std::sync::Mutex<()>>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));
pub(crate) fn ledger() -> std::sync::MutexGuard<'static, Ledger> {
    LEDGER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
/// Persistent physical + external charge bound to the same backing/stream
/// lifetime as the provider. Graphs retain it through Arc<NativeBf16GemmEx>.
/// Unknown retains only this reservation, not an initializer lock.
pub(crate) struct Admission {
    owner: Arc<CudaStream>,
    family: ProviderFamily,
}
impl Admission {
    pub(crate) fn prepare<T>(
        stream: &CudaStream,
        family: ProviderFamily,
        charge: Charge,
        prepare: impl FnOnce(Self) -> Result<T>,
    ) -> Result<T> {
        if unsafe { ferrule_bf16_gemmex_process_config_valid() } != 1 {
            return Err(invalid(
                "set CUBLAS_WORKSPACE_CONFIG=:0:0 in the process launcher BEFORE any cuBLAS initialization; restart this process (preparation never mutates environment)",
            ));
        }
        cu(stream.context().check_graph_ready())?;
        let device = stream.context().ordinal();
        let initialization = INITIALIZERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(device)
            .or_insert_with(|| Arc::new(std::sync::Mutex::new(())))
            .clone();
        // Coordinate only provider preparation on this device. This cannot stop
        // unrelated CUDA allocations, nor make global free-memory deltas attributable.
        let _guard = initialization
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let owner = cu(stream.owner())?;
        let key = Arc::as_ptr(&owner) as usize;
        ledger().reserve((key, family), device, charge)?;
        let admission = Self { owner, family };
        cu(stream.synchronize())?;
        let (free, _) = cu(stream.context().memory_info())?;
        // Free bytes already exclude live backing allocations, but not unused
        // library reservations. Conservatively subtract all admitted external
        // reserves on this device without using a measured delta as accounting.
        let outstanding_external = ledger().devices[&device].external;
        let demand = charge
            .storage_grant
            .checked_add(outstanding_external)
            .ok_or_else(|| invalid("storage/external headroom overflow"))?;
        if free < demand {
            return Err(invalid(format!(
                "owner admission needs {demand} bytes (new physical grant + admitted external headroom), free={free}; unrelated allocations are not serialized"
            )));
        }
        prepare(admission)
    }
}
impl Drop for Admission {
    fn drop(&mut self) {
        if self.owner.context().graph_completion_unknown() {
            std::mem::forget(Arc::clone(&self.owner));
            return;
        }
        ledger().release((Arc::as_ptr(&self.owner) as usize, self.family));
    }
}
