# Localhost CUDA-linear acceptance entry

`scripts/validate_network_localhost.py` is an opt-in Linux acceptance tool for
real `ferrule rank-serve` endpoints and `ferrule bench-parallel` controllers.
It is not model-serving validation, a multi-host test, or a CI default.

Safe discovery (no listener, model, CUDA initialization, or child process):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
  /opt/conda310/bin/python -B scripts/validate_network_localhost.py --help
```

Execution requires an already-built CUDA CLI binary, two distinct owner-assigned
physical GPU ordinals via `--devices`, a **new** `--output` directory, and the
explicit `--approve-linear-gpu` flag. Do not run it during another GPU owner's
work. It does not compile or install anything. It generates temporary test-only
TLS credentials with local OpenSSL, binds loopback endpoints, and removes the
credentials when the run exits normally through its cleanup path.

The ten cases cover healthy DP/column-TP/row-TP, server/client certificate pins,
CA/SAN rejection, a stopped or killed child, and controller disconnect. The
controller and rank endpoints must coexist: they are not independent sequential
jobs. Faults intentionally signal the observed rank child or controller.

## Current supervision limit

This script retains its existing local `Run` process groups, background `/proc`
observer, `prctl` subreaper, and bounded cleanup. Its `children_absent` field is an
observation of recorded PID/start-time pairs, **not** an authenticated
`test_vllm_qwen35.run_process` handshake plus kernel `ECHILD` drain proof. It also
does not establish post-run whole-host GPU quiescence. Neither the absence of
observed children nor successful CLI exits should be relabeled as those proofs.

The next supervision change should put one complete multi-process case under
`scripts/test_vllm_qwen35.py`'s existing authenticated custodian, preserving
concurrent endpoints, fault timing, owned signalling, output checks, deadlines,
and failure receipts. Replacing each `Popen` with a blocking sequential call
would break the test. That migration is deliberately deferred, not implemented
as another supervisor framework.
