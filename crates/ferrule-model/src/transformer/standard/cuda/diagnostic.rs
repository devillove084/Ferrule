//! Borrowed, opt-in observations at existing operator boundaries. No alternate forward.
use super::*;
use std::cell::{Cell, RefCell};

/// Buffers remain owned by the original operation. Observers may synchronously
/// read them through the owner, but cannot mutate or retain borrowed operands.
pub struct CudaDiagnosticEvent<'a> {
    pub layer: Option<usize>,
    pub name: &'static str,
    pub operators: &'a CudaOperators,
    pub buffer: &'a CudaF32Buffer,
    pub shape: RowsShape,
    pub linear: Option<&'a PreparedLinear>,
    /// Available for both prepared and payload-free resident expert linears.
    pub parameter: Option<&'a BoundParameter>,
    pub norm: Option<&'a PreparedNorm>,
    pub routes: Option<&'a RouterRoutes>,
}
type Callback = dyn FnMut(&CudaDiagnosticEvent<'_>) -> Result<()>;
#[derive(Default)]
pub(super) struct DiagnosticTrace {
    pub layer: Cell<Option<usize>>,
    callback: RefCell<Option<Box<Callback>>>,
}
impl CudaStandardDecoderOperators {
    /// Disabled by default: no D2H, allocation, or operand retention. An installed
    /// observer explicitly chooses its readback/storage policy. It must not
    /// re-enter this owner. Observer errors follow normal forward failure custody.
    pub fn set_diagnostic_trace<F>(&self, callback: F)
    where
        F: FnMut(&CudaDiagnosticEvent<'_>) -> Result<()> + 'static,
    {
        *self.diagnostic.callback.borrow_mut() = Some(Box::new(callback));
    }
    pub fn clear_diagnostic_trace(&self) {
        self.diagnostic.callback.borrow_mut().take();
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn trace_buffer(
        &self,
        name: &'static str,
        buffer: &CudaF32Buffer,
        shape: RowsShape,
        linear: Option<&PreparedLinear>,
        norm: Option<&PreparedNorm>,
        routes: Option<&RouterRoutes>,
    ) -> Result<()> {
        if let Some(callback) = self.diagnostic.callback.borrow_mut().as_mut() {
            callback(&CudaDiagnosticEvent {
                layer: self.diagnostic.layer.get(),
                name,
                operators: &self.ops,
                buffer,
                shape,
                linear,
                parameter: linear
                    .map(|p| p.parameter().binding())
                    .or_else(|| norm.map(|p| p.parameter().binding())),
                norm,
                routes,
            })?;
        }
        Ok(())
    }
    pub(super) fn trace_expert_linear(
        &self,
        name: &'static str,
        buffer: &CudaF32Buffer,
        shape: RowsShape,
        parameter: &BoundParameter,
    ) -> Result<()> {
        if let Some(callback) = self.diagnostic.callback.borrow_mut().as_mut() {
            callback(&CudaDiagnosticEvent {
                layer: self.diagnostic.layer.get(),
                name,
                operators: &self.ops,
                buffer,
                shape,
                parameter: Some(parameter),
                linear: None,
                norm: None,
                routes: None,
            })?;
        }
        Ok(())
    }

    pub(super) fn trace_rows(&self, name: &'static str, rows: &Rows) -> Result<()> {
        if self.diagnostic.callback.borrow().is_none() {
            return Ok(());
        }
        self.trace_buffer(name, self.f32(rows)?, rows.shape(), None, None, None)
    }
}
