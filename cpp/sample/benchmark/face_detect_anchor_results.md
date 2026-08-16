# Face detector anchor generation optimization 2/6

Date: 2026-08-17

## Scope

- Before: build a temporary vector of anchor-center coordinates for every stride and every detection call.
- After: traverse the same `y -> x -> anchor` order inside `_decode()` and use each center directly.
- Removed per-call temporary storage: 2,100 floats at input 160, 8,400 at input 320, and 33,600 at input 640.
- Benchmark: Release, 50 timed iterations after five warmups, using the four sample images at detector sizes 160, 320, and 640.

## Correctness

All 12 cases passed before and after. Face counts, rectangle/confidence digests, repeated-run digests, and timed-run digests were identical.

## Timing

The values below average the four image cases at each detector size. Positive delta means the optimized build was faster.

| Detector | Before mean / p50 | After mean / p50 | Mean delta |
|---:|---:|---:|---:|
| 160 | 2.642 / 2.354 ms | 2.806 / 2.347 ms | -6.19% |
| 320 | 10.235 / 10.070 ms | 8.856 / 8.780 ms | +13.47% |
| 640 | 32.178 / 31.818 ms | 27.762 / 27.621 ms | +13.72% |
| Fixed 12-case aggregate | 180.220 / 176.968 ms | 157.697 / 154.989 ms | +12.50% |

The 160 mean contains an isolated multi-person-image outlier; its p50 changed by only +0.30%. End-to-end timing is sensitive to CPU state, so the timing result is observational rather than a guarantee. The correctness result and removal of the temporary allocations are deterministic.
