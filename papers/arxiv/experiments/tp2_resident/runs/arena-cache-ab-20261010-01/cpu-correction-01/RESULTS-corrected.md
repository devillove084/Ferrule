# CPU closeout correction (historical GPU run, no new GPU work)

Supersedes only sample bookkeeping and run-window wording. Original owner-closeout and RESULTS remain byte-identical.
Original owner-closeout SHA256: `918ced0f852b36c01b6d3616b2b9fbe30e4421305df9d8cbb7352bf7f5d4d8bb`. Original RESULTS SHA256: `d571acaed8b9a003d7fbf40b6376d05ce7b5dccf96521a52db516ecdeb1e87b4`.

| Lifecycle | kind=sample | GPU present | GPU missing | excluded monitor_terminal |
|---|---:|---:|---:|---:|
| S1 | 161 | 161 | 0 | 1 |
| S2 | 159 | 159 | 0 | 1 |
| OFF | 113 | 113 | 0 | 1 |
| ON | 162 | 162 | 0 | 1 |

## Throughput denominator clarification (rates unchanged)

The recorded `wall_seconds` is the **actual runner-observed fixed-work run window**: first measured HTTP send to the runner timestamp after terminal/progress checkpoint and executor exit. It is not simply the last request EOF/client terminal span. Startup and warmups are excluded.

- OFF: 7.510810526088 s; tail after last request client_observed_at = 31.745991 ms (after ended_at = 31.898541 ms). Original output rate remains 5.325656912934 tok/s.
- ON: 15.954496456310 s; tail after last request client_observed_at = 34.720764 ms (after ended_at = 34.840874 ms). Original output rate remains 2.507130206806 tok/s.

Tail components are not individually timed; no adjusted rate replaces the old result. All original performance metrics, source pins and limitations remain historical and unchanged.

No current production/source/quality validation or GPU/proc/lock query was performed. Final idle/release fields refer to the historical receipts, not a new observation. NVML host/local PID attribution remains unavailable; no proof of absence of foreign interference. Null activity does not prove zero decode; 54,621 evictions per ON owner remain whole-lifecycle, not measured/hot-phase counters. Same flags with different sources remains non-isolated and non-statistical.
