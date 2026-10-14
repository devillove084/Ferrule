State is not implicitly cloneable or transferable to another owner:

```compile_fail
use ferrule_runtime::parallel::pipeline::PipelineStateProgram;
fn copy<P: PipelineStateProgram>(state: &P::SequenceState) -> P::SequenceState {
    state.clone()
}
```

```compile_fail
use ferrule_runtime::parallel::pipeline::PipelineStateProgram;
fn transfer<P: PipelineStateProgram>(state: P::SequenceState) {
    std::thread::spawn(move || drop(state));
}
```
