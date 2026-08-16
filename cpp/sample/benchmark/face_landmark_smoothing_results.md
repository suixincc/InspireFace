# Face landmark smoothing optimization 6/6

Date: 2026-08-17

## Scope

- Before: every smoothing call copied the complete current landmark vector before processing, then copied the newly built smoothed frame into the history queue.
- After: each point's current coordinates are read once before that point is updated, the history frame reserves its final capacity, and the completed frame is moved into the queue.
- Unchanged behavior: smoothing activation, distance and weight formulas, floating-point operation order, point update order, stored-history contents, history eviction order, and all C API tracking output.

The history intentionally stores the smoothed landmark frame, not the original detector frame. The optimization preserves that behavior exactly.

## Test design

The sample benchmark has two layers.

The direct internal test feeds 257 deterministic but different generated landmark frames through `DynamicSmoothParamUpdate()`. It hashes all 106 returned points and every point in the complete history after each call.

| Scenario | History frames | Smooth ratio | Purpose |
|---|---:|---:|---|
| `smooth_n1_h0` | 1 | 0.00 | Small-history and zero-ratio boundary |
| `smooth_n5_default` | 5 | 0.05 | Production-like default path |
| `smooth_n8_h020` | 8 | 0.20 | Larger history and stronger smoothing |

The end-to-end test uses the public C API in `HF_DETECT_MODE_LIGHT_TRACK` with different images and settings.

| Scenario | Image | Detector / interval | Smoothing | Expected faces |
|---|---|---:|---:|---:|
| `frontal_n1` | `kun.jpg` | 320 / every 5 frames | 1 / 0.00 | exactly 1 |
| `profile_n5` | `jntm.jpg` | 320 / every 5 frames | 5 / 0.05 | exactly 1 |
| `woman_n8` | `woman.png` | 320 / every 5 frames | 8 / 0.20 | exactly 1 |
| `multi_n5` | `pedestrian.png` | 640 / every frame | 5 / 0.05 | 16 to 20 |

Every tracked frame validates face counts, positive rectangles, unique and stable track IDs, nonnegative and nondecreasing track counts, finite confidence and pose, complete serialized token bytes, five key points, and all 106 dense landmarks.

Formal command:

```text
FaceLandmarkSmoothingBenchmarkSample test_res/pack/Pikachu test_res 50 8 10000
```

Independent before and optimized binaries were run three times each in a balanced alternating order. Every scenario therefore used 10,000 directly timed calls plus one initial, eight warmup, and 50 timed end-to-end frames per formal run.

## Correctness and consistency

All six formal runs passed. Each direct sequence digest was identical in every repetition and matched exactly before and after:

| Scenario | Sequence digest |
|---|---:|
| `smooth_n1_h0` | `0xf1a13098e7cb2b28` |
| `smooth_n5_default` | `0xe11b4405b0572aa8` |
| `smooth_n8_h020` | `0x5363feffda93413c` |

The complete end-to-end sequence digests also matched exactly in all six formal runs:

| Scenario | Observed faces | Sequence digest |
|---|---:|---:|
| `frontal_n1` | 1 | `0x902692aee2b9177b` |
| `profile_n5` | 1 | `0x972a2d484a542b38` |
| `woman_n8` | 1 | `0xe8785cec1ac222c2` |
| `multi_n5` | 16 to 17 | `0xc835bcd94d730be8` |

This gives an exact check of the smoothing outputs and history state, followed by an exact check of all relevant observable detection and tracking results through the C API.

## Direct smoothing latency

The table uses the median result from three independent formal runs. Positive delta means the optimized implementation was faster. Values are microseconds per smoothing call.

| Scenario | Before mean / p50 / p95 | After mean / p50 / p95 | Mean delta |
|---|---:|---:|---:|
| History 1, ratio 0.00 | 8.499 / 7.333 / 13.833 | 3.818 / 3.334 / 6.461 | +55.08% |
| History 5, ratio 0.05 | 13.789 / 12.375 / 21.294 | 8.346 / 7.334 / 13.125 | +39.47% |
| History 8, ratio 0.20 | 18.256 / 16.292 / 27.625 | 11.889 / 10.667 / 17.958 | +34.88% |

Mean, p50, and p95 improved in all three direct scenarios.

## End-to-end latency

These values are milliseconds per `HFExecuteFaceTrack()` call and use the median result from the same three runs.

| Scenario | Before mean / p50 / p95 | After mean / p50 / p95 | Mean delta |
|---|---:|---:|---:|
| Frontal, detect every 5 frames | 2.157 / 0.768 / 7.699 | 2.107 / 0.784 / 7.889 | +2.29% |
| Profile, detect every 5 frames | 2.212 / 0.857 / 7.761 | 2.186 / 0.811 / 8.193 | +1.16% |
| Woman, detect every 5 frames | 2.185 / 0.839 / 7.746 | 2.145 / 0.835 / 7.653 | +1.85% |
| Multi-person, detect every frame | 38.587 / 38.298 / 43.130 | 37.917 / 37.741 / 40.555 | +1.74% |

The direct optimization saves only several microseconds inside a much larger detector/tracker call, so mixed single-image p50 and p95 movements are expected timing noise. The median mean did not regress in any scenario, and the multi-person case improved in mean, p50, and p95. The direct benchmark is the authoritative measurement for this isolated change.

## Conclusion

The optimization removes one full temporary landmark copy and one history-frame copy per call. Direct smoothing outputs and history state remain bit-for-bit identical, all tested C API detection and tracking outputs remain bit-for-bit identical, direct latency improves across every tested history size, and no end-to-end mean regression was measured.
