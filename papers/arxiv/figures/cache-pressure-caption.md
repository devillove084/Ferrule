# Cache-pressure figure

Source: `cache-pressure-budget.py`  
Vector output: `cache-pressure-budget.pdf`  
Raster preview: `cache-pressure-budget.png`

## Caption

**Coupled cache-budget sensitivity without observed read reduction.**
Three fresh, fixed-order blocking lifecycles jointly vary requested expert
entries and cache byte ceilings: 8/64 MiB, 64/320 MiB and 128/640 MiB. **(A)**
Ending logical cache charge and owner logical high-water marks increase under
the dashed requested ceilings; the two bars and ceiling are parallel values,
not additive components. Charge includes 8 MiB scratch. **(B)** Selected-device
GPU-memory sampled maxima increase from 3,997 to 4,125 to 4,317 MiB. Each value
is the maximum of 36 observations, not an exact lifetime peak or proven upper
bound; attribution relies on exclusive operator allocation, not proven NVML
host/local PID translation. **(C)** Logical checkpoint reader bytes remain
36,240,064,128 at every point, with zero expert hits, 69,114 read calls and
11,519 verified expert reads per point. These are not physical NAS backend
bytes. **(D)** Individual client HTTP end-to-end times for four distinct
prompts are shown as dots, n=1 per prompt/point, without confidence intervals
or pooling. The approximately three three-second candidate-pin checks per
request are retained outside HTTP E2E in the measured helper/request-phase
evidence, not removed from total elapsed time.

All 12 HTTP requests complete; each point has three exact eight-ID canaries
and one two-token length/usage smoke, giving 78 tokens total (72 oracle-covered).
Requested ceilings, observed logical charge and sampled GPU memory are not
physical-admission measurements or additive partitions. This descriptive
coupled two-knob sweep does not isolate an LRU effect, establish a causal
speedup, demonstrate working-set fit, or locate a physical capacity frontier.
The older censored/admission-rejected studies are retained separately and
never pooled. Full-logit/route/state numerical quality at the new pressure
points remains unverified; the visible HTTP/ID scope does not transfer the
prior middle-only numerical qualification.

## Regeneration

From the repository root, with Matplotlib 3.8.4 available:

```sh
python3 -B papers/arxiv/figures/cache-pressure-budget.py
```

The script verifies and derives the points directly from the public typed
records at
`papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/`.
It writes only these two new outputs, uses an offscreen CPU renderer, and does
not modify manuscript or existing figure build files.
