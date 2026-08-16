# Face tracker NMS optimization 4/6

Date: 2026-08-17

## Scope

- Before: the sort comparator copied two complete `FaceObjectInternal` objects for every comparison, and suppression erased elements from the face and area vectors inside the nested loop.
- After: the comparator takes const references, candidates are compared with an ordered retained-face list, suppressed candidates stop at the first match, and retained objects are moved once into preallocated storage.
- Unchanged behavior: descending confidence order, IoU formula, threshold comparison, greedy suppression priority, and retained-face order.

## Test design

The sample benchmark runs the public C API in `HF_DETECT_MODE_LIGHT_TRACK` so the production tracker and its NMS implementation are exercised end to end.

| Scenario | Image | Detector input | Detect interval | Expected faces |
|---|---|---:|---:|---:|
| `single_static` | `kun.jpg` | 320 | 5 | exactly 1 |
| `multi_static` | `pedestrian.png` | 640 | 1 | 16 to 20 |
| `no_face_static` | `view.jpg` | 320 | 1 | exactly 0 |

Every frame validates the result count, positive rectangles, finite confidence and pose angles, unique track IDs, nonnegative/nondecreasing track counts, stable IDs, and five finite key points extracted from every face token. The sequence digest covers all of those observable values.

## Correctness and consistency

- Before: three independent runs, each with one initial frame, five warmups, and 50 timed frames per scenario; all passed.
- After: the same three-run protocol and parameters; all passed.
- Each scenario produced the same sequence digest in all six formal runs, including exact before/after matches:
  - `single_static`: `0x75d173e8595f5524`
  - `multi_static`: `0x53790c2c48afbdd3`
  - `no_face_static`: `0x792deb822ade6003`
- Observed face ranges also matched: 1, 16 to 17, and 0 respectively.

## Latency

The table uses the median result from three independent formal runs. Positive delta means the optimized build was faster. Values are end-to-end tracker latency in milliseconds, not isolated NMS time.

| Scenario | Before mean / p50 / p95 | After mean / p50 / p95 | Mean delta |
|---|---:|---:|---:|
| Single, detect every 5 frames | 2.045 / 0.732 / 7.461 | 2.028 / 0.750 / 7.415 | +0.85% |
| Multi-person, detect every frame | 37.191 / 36.955 / 39.940 | 36.720 / 36.476 / 39.051 | +1.27% |
| No face, detect every frame | 5.963 / 5.766 / 7.010 | 5.829 / 5.610 / 7.349 | +2.23% |

The multi-person case is the most representative stress case for this NMS path because it processes 16 to 17 tracked faces and invokes detection every frame. Its mean, p50, and p95 all improved by 1.27%, 1.29%, and 2.23% respectively. The mixed movements in the single-person and no-face tails are consistent with normal end-to-end timing noise.

## Conclusion

All observable tracking and detection results remained bit-for-bit consistent. The implementation removes expensive comparator copies and repeated vector erases, while the multi-person stress case shows a small end-to-end latency improvement. As with all whole-pipeline measurements, the percentage is observational rather than a performance guarantee.
