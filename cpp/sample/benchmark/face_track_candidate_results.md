# Face tracker candidate-transfer optimization 5/6

Date: 2026-08-17

## Scope

- Before: each detected `FaceObjectInternal` was copied into `candidate_faces_` and copied again when appended to `trackingFace`.
- After: candidates are constructed in place, both candidate and destination vectors reserve their required capacity, and candidates are moved into `trackingFace` in their original order.
- The normal detection path and track-lost recovery path now share the same append helper.
- Unchanged behavior: candidate staging, face filtering, maximum-face handling, tracking ID assignment, append order, tracking execution, serialization timing, C API structures, and token format.

`FaceObjectInternal` owns several vectors containing landmarks and temporal state. Moving transfers those buffers instead of allocating and copying their elements. No object is moved after `FaceSession` creates its C API caches.

## C API cache safety

`HFExecuteFaceTrack()` returns session-owned arrays and tokens produced after `FaceTrackModule::UpdateStream()` completes. Each token is a serialized value-only `FaceTrackWrap`, not a pointer or reference to `FaceObjectInternal`.

The benchmark explicitly verifies:

- rectangles, confidence, pose, track IDs, and track counts;
- complete serialized token bytes;
- five key points and 106 dense landmarks decoded from every token;
- a token copied with `HFCopyFaceBasicToken()`, used after a subsequent `HFExecuteFaceTrack()` call;
- quality and recognition feature extraction from that copied token;
- interaction and emotion pipeline processing, including the `inGroupIndex + trackId` link back to the current internal tracking object.

Raw token pointers returned directly by `HFExecuteFaceTrack()` remain session-owned and are invalidated by the next tracking call. Callers that retain a token across calls must continue to use `HFCopyFaceBasicToken()`; this existing lifetime rule is unchanged.

## Test matrix

| Scenario | Mode | Image | Detector / interval | Purpose |
|---|---|---|---:|---|
| `always_multi` | Always Detect | `pedestrian.png` | 640 / every frame | Construct and transfer 16 to 20 heavy candidates every frame |
| `light_single_cache` | Light Track | `kun.jpg` | 320 / every 5 frames | Stable tracking plus all dependent C API cache and token checks |
| `light_multi_redetect` | Light Track | `pedestrian.png` | 640 / every frame | Existing tracks plus repeated detection and candidate merging |
| `track_by_detect_single` | Track By Detection | `kun.jpg` | 320 / every frame | ByteTrack candidate-construction branch |
| `always_no_face` | Always Detect | `view.jpg` | 320 / every frame | Empty-candidate boundary and timing control |

Each formal before/after run used two initial frames, five warmups, and 50 timed frames per scenario. Three independent runs were recorded for both implementations.

## Correctness and consistency

All 30 formal scenario runs passed. Every sequence digest matched across all three repetitions and matched exactly before and after:

| Scenario | Faces | Sequence digest |
|---|---:|---:|
| `always_multi` | 17 | `0xd056b5fb72020e59` |
| `light_single_cache` | 1 | `0xa6727d771d88b44d` |
| `light_multi_redetect` | 17 on the first two frames | `0x42ea0b51902d84ce` |
| `track_by_detect_single` | 1 | `0x79162a4be88f6e57` |
| `always_no_face` | 0 | `0x8e391450d0acedea` |

The copied-token dependent digest was `0x3cc3df0d7a7b5492`, and the interaction/emotion pipeline digest was `0xc6736821415863b3` in every before/after run.

## Latency

The table uses the median result from three independent runs. Positive delta means the optimized build was faster. Values are end-to-end C API latency in milliseconds, not isolated candidate-transfer time.

| Scenario | Before mean / p50 / p95 | After mean / p50 / p95 | Mean delta |
|---|---:|---:|---:|
| Always Detect, multi-person | 38.526 / 37.858 / 43.587 | 36.612 / 36.507 / 39.186 | +4.97% |
| Light Track, single/cache | 2.000 / 0.753 / 7.304 | 1.981 / 0.717 / 7.245 | +0.92% |
| Light Track, multi/redetect | 38.000 / 37.270 / 43.486 | 36.282 / 36.254 / 38.997 | +4.52% |
| Track By Detection, single | 7.066 / 6.886 / 8.799 | 6.735 / 6.471 / 8.410 | +4.68% |
| No-face control | 6.277 / 6.008 / 8.357 | 5.994 / 5.816 / 7.598 | +4.51% |

The relevant face scenarios did not regress, but the no-face control improved by a similar amount in this measurement window. Therefore the end-to-end percentage cannot be attributed entirely to candidate moves and should be treated as timing noise plus a possible small improvement. The deterministic benefit is removal of repeated deep copies and avoidable allocations.

## Conclusion

The candidate-transfer optimization preserves all observable detection, tracking, serialized-token, copied-token, feature, and pipeline results exactly. It does not change the C API cache or token lifetime model. The code removes heavyweight copies without a measured end-to-end regression; no guaranteed speedup is claimed from this whole-pipeline benchmark.
