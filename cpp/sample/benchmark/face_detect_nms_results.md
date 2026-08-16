# Face detector NMS optimization 3/6

Date: 2026-08-17

## Scope

- Before: erase suppressed faces and their areas from two vectors inside the NMS inner loop.
- Rejected prototype: mark suppressed candidates and compact at the end. It preserved results but consistently increased end-to-end latency.
- Selected implementation: greedy NMS over score-sorted candidates, comparing each candidate only with retained faces and stopping at the first suppressing face.
- Unchanged behavior: score ordering, IoU formula, NMS threshold, greedy suppression priority, and final area ordering.

## Correctness and accuracy

The benchmark digest covers face count, rectangles, detection confidence, roll/yaw/pitch, and five key points extracted from each face token.

- Before: three runs, 12 image/size cases per run, all passed.
- After: three runs using the same cases and parameters, all passed.
- Every before/after correctness digest and timed-run digest matched exactly.
- Expected detection counts passed for frontal, profile, raised-head, and multi-person images at detector sizes 160, 320, and 640.
- A final no-face edge case was added and passed at all three detector sizes, producing exactly zero faces.

## Latency

Each formal run used five warmups and 50 timed iterations per case. Values below are calculated from the median result of three independent runs, averaged across the four common images at each detector size. Positive delta means the selected implementation was faster.

| Detector | Before mean / p50 | After mean / p50 | Mean delta |
|---:|---:|---:|---:|
| 160 | 2.411 / 2.253 ms | 2.472 / 2.334 ms | -2.52% |
| 320 | 8.764 / 8.560 ms | 8.578 / 8.422 ms | +2.12% |
| 640 | 27.588 / 27.507 ms | 27.366 / 27.169 ms | +0.80% |
| Fixed 12-case aggregate | 155.054 / 153.280 ms | 153.665 / 151.700 ms | +0.90% |

The 640 multi-person case changed from 36.909 / 36.841 ms to 36.744 / 36.529 ms, a mean improvement of 0.45% and a p50 improvement of 0.85%.

## Conclusion

Detection accuracy and complete observable results are unchanged. The selected NMS implementation removes repeated vector erases without the latency regression seen in the rejected suppression-marker prototype. The measured end-to-end improvement is small and should be treated as noise-level rather than a guaranteed speedup.
