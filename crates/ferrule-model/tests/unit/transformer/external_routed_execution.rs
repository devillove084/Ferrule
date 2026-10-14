//! Exercise the real FFN dispatch boundary with synchronous arithmetic and an
//! instrumented producer, so ordering/failure checks also run without a GPU.
use super::*;
use crate::transformer::{RouterRoutes, RowsArenaId};
use std::cell::RefCell;
use std::rc::Rc;

type Trace = Rc<RefCell<Vec<&'static str>>>;

struct Producer {
    cpu: CpuStandardDecoderOperators,
    trace: Trace,
    fail_boundary: bool,
}

macro_rules! delegate {
    ($name:ident($($arg:ident: $ty:ty),*) -> $out:ty) => {
        fn $name(&mut self, $($arg: $ty),*) -> Result<$out> {
            self.trace.borrow_mut().push(stringify!($name));
            self.cpu.$name($($arg),*)
        }
    };
}

impl StandardDecoderOperators for Producer {
    fn backend_name(&self) -> &'static str {
        self.cpu.backend_name()
    }
    fn precision(&self) -> ExecutionPrecisionPolicy {
        self.cpu.precision()
    }
    fn before_external_routed_execution(&mut self) -> Result<()> {
        self.trace.borrow_mut().push("barrier");
        if self.fail_boundary {
            Err(model_error("lost producer proof"))
        } else {
            Ok(())
        }
    }
    delegate!(bind_rows(rows: Rows) -> Rows);
    delegate!(embedding(embedding: &PreparedEmbedding, ids: &[u32], arena: Option<RowsArenaId>) -> OperatorProgress<Rows>);
    delegate!(linear(linear: &PreparedLinear, input: &Rows, arena: Option<RowsArenaId>) -> OperatorProgress<Rows>);
    delegate!(rms_norm(norm: &PreparedNorm, input: &Rows, heads: usize, arena: Option<RowsArenaId>) -> OperatorProgress<Rows>);
    delegate!(rope(descriptor: &RotaryEmbedding, table: &PreparedRope, input: Rows, heads: usize, positions: &[usize]) -> OperatorProgress<Rows>);
    delegate!(paged_gqa(kv: &mut dyn KvView, request: GqaRequest<'_>) -> OperatorProgress<Rows>);
    delegate!(router(logits: &Rows, policy: &MoeRouterSpec) -> OperatorProgress<RouterRoutes>);
    delegate!(dense_swiglu(expert: &PreparedSwiGlu, input: &Rows, arena: Option<RowsArenaId>) -> OperatorProgress<Rows>);
    delegate!(routed_swiglu(layer: usize, input: &Rows, routes: &RouterRoutes, experts: &mut dyn ExpertProvider, arena: Option<RowsArenaId>) -> OperatorProgress<Rows>);
    delegate!(residual(residual: Rows, update: &Rows) -> OperatorProgress<Rows>);
    delegate!(lm_head(head: &PreparedLinear, input: &Rows, arena: Option<RowsArenaId>) -> OperatorProgress<Rows>);
    delegate!(shared_expert_gate(input: Rows, gate: &Rows) -> OperatorProgress<Rows>);
}

struct Executor {
    trace: Trace,
}
impl RoutedSwiGluExecutor for Executor {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        assert_eq!(self.trace.borrow().last(), Some(&"barrier"));
        self.trace.borrow_mut().push("executor");
        assert_eq!(request.context.layer, 0);
        assert_eq!(request.context.source_rank, ParallelRankId::new(0));
        assert_eq!(request.sequences, &[10, 10, 11, 11]);
        assert_eq!(request.routes.rows(), request.input.shape().rows());
        check_active(request.context.transaction)?;
        Ok(OperatorProgress::Ready(Rows::Host(HostRows::new(
            request.input.shape(),
            RowsDType::F32,
            request.arena,
            vec![0.; request.input.shape().elements()],
        )?)))
    }
}

#[test]
fn external_routed_boundary_precedes_executor_and_liveness_and_fails_closed() {
    for failure in 0..3 {
        let f = Fixture::new(false);
        let input = rows(&oracle()["input"]);
        let trace = Trace::default();
        let mut ops = Producer {
            cpu: CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32()),
            trace: Rc::clone(&trace),
            fail_boundary: failure == 1,
        };
        let mut executor = Executor {
            trace: Rc::clone(&trace),
        };
        let transaction = ExecutionTransactionId::new(7).unwrap();
        let mut check = |id| {
            assert_eq!(id, transaction);
            assert_eq!(trace.borrow().last(), Some(&"executor"));
            trace.borrow_mut().push("check_active");
            if failure == 2 {
                Err(model_error("cancelled"))
            } else {
                Ok(())
            }
        };
        let mut execution = RoutedLayerExecution {
            transaction,
            source_rank: ParallelRankId::new(0),
            sequences: &[10, 10, 11, 11],
            executor: &mut executor,
            check_active: &mut check,
        };
        let result = execute_feed_forward(
            &mut ops,
            &f.materializer,
            0,
            &f.block.kind,
            &input,
            Some(&mut execution),
        );
        let mut expected = vec![
            "linear",
            "router",
            "dense_swiglu",
            "linear",
            "shared_expert_gate",
            "barrier",
        ];
        match failure {
            0 => {
                expected.extend(["executor", "check_active", "bind_rows", "residual"]);
                assert_eq!(result.unwrap().shape(), input.shape());
            }
            1 => assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("lost producer proof")
            ),
            _ => {
                expected.extend(["executor", "check_active"]);
                assert!(result.unwrap_err().to_string().contains("cancelled"));
            }
        }
        assert_eq!(*trace.borrow(), expected);
    }
}

#[test]
fn local_routed_execution_does_not_invoke_the_external_boundary() {
    let f = Fixture::new(false);
    let input = rows(&oracle()["input"]);
    let trace = Trace::default();
    let mut ops = Producer {
        cpu: CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32()),
        trace: Rc::clone(&trace),
        fail_boundary: true,
    };
    let result =
        execute_feed_forward(&mut ops, &f.materializer, 0, &f.block.kind, &input, None).unwrap();
    compare(&result, &oracle()["output"]);
    assert!(!trace.borrow().contains(&"barrier"));
    assert!(trace.borrow().contains(&"routed_swiglu"));
    // Existing providers need not implement the new hook.
    ops.cpu.before_external_routed_execution().unwrap();
}
