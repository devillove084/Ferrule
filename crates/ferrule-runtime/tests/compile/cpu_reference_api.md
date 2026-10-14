## Default-build fake-provider boundary

Unit-test support is not a default dependency API.

```compile_fail
use ferrule_runtime::io::testing::FakeMaterializationProvider;
```

```compile_fail
use ferrule_runtime::scheduling::PhysicalResourceBroker;
let _ = PhysicalResourceBroker::testing_default();
```

The explicit hybrid CPU owner factory and its configuration/program types are
not available from a default production dependency.

```compile_fail
use ferrule_runtime::parallel::pipeline::HybridCpuTensorConfig;
```

```compile_fail
use ferrule_runtime::parallel::pipeline::HybridCpuTensorPipelineStageProgram;
```

```compile_fail
use ferrule_runtime::parallel::pipeline::HybridCpuTensorStage;
```

```compile_fail
use ferrule_runtime::parallel::pipeline::PipelineParallelExecutor;
use ferrule_runtime::parallel::PipelineRank;
use ferrule_model::transformer::BoundDecoderResources;
let _ = PipelineParallelExecutor::new_hybrid_cpu_tensor::<
    fn(PipelineRank) -> ferrule_common::Result<BoundDecoderResources>
>;
```
