# Face detector post-processing optimization 1/6

Date: 2026-08-17

## Scope

- Before: `FaceDetectAdapt::_decode()` sorts the accumulated candidates by face area after every anchor.
- After: remove that repeated sort and its unused comparator. `_nms()` still sorts by confidence, and the final result is still sorted by face area.
- Benchmark type: end-to-end face detection through the stable C API, exercising the changed internal path.
- Build: Release, macOS 15.6.1, arm64, MNN CPU-only, InspireCV CPU, Pikachu-t4.0 model pack.
- Inputs: frontal face, profile face, raised-head face, and a multi-person image at detector sizes 160, 320, and 640.

Build and run:

```sh
cmake -S . -B build_probe \
  -DCMAKE_BUILD_TYPE=Release \
  -DISF_BUILD_WITH_TEST=OFF \
  -DISF_BUILD_WITH_SAMPLE=ON
cmake --build build_probe --target FaceDetectPostprocessBenchmarkSample -j4
./build_probe/sample/benchmark/FaceDetectPostprocessBenchmarkSample \
  test_res/pack/Pikachu test_res 100 5
```

## Expected-result comparison

All 12 cases passed before and after. Each digest was stable across the initial run, three correctness repeats, and every timed iteration. The before/after face counts and digests were identical.

| Detector | Image | Faces | Expected | Digest |
|---:|---|---:|---:|---|
| 160 | frontal | 1 | 1 | `0x736139dd3010628b` |
| 160 | profile | 1 | 1 | `0xee17ad8c1bf3b986` |
| 160 | raised head | 1 | 1 | `0xd1bd53c78a478db4` |
| 160 | pedestrian | 1 | 1-6 | `0x6fd27440c42092ed` |
| 320 | frontal | 1 | 1 | `0x8b8b173fed00a30c` |
| 320 | profile | 1 | 1 | `0xb8c1951f14797671` |
| 320 | raised head | 1 | 1 | `0x1b982364f9fda66b` |
| 320 | pedestrian | 10 | 10-11 | `0x72844b885962768c` |
| 640 | frontal | 1 | 1 | `0x3bcdb853af2f6400` |
| 640 | profile | 1 | 1 | `0xc404974824963b19` |
| 640 | raised head | 1 | 1 | `0x1291d23e5df999fe` |
| 640 | pedestrian | 17 | 16-20 | `0x20654014da592869` |

## Timing comparison

The table is the 100-iteration confirmation run after five warmups. Positive delta means the optimized build was faster. Times are end-to-end milliseconds per call.

| Detector | Image | Before mean / p50 | After mean / p50 | Mean delta |
|---:|---|---:|---:|---:|
| 160 | frontal | 2.692 / 2.488 | 2.707 / 2.591 | -0.58% |
| 160 | profile | 2.878 / 2.726 | 2.852 / 2.760 | +0.92% |
| 160 | raised head | 2.960 / 2.884 | 2.689 / 2.672 | +9.16% |
| 160 | pedestrian | 2.484 / 2.336 | 2.391 / 2.257 | +3.74% |
| 320 | frontal | 6.824 / 6.723 | 6.984 / 6.726 | -2.36% |
| 320 | profile | 7.215 / 6.846 | 7.252 / 6.801 | -0.51% |
| 320 | raised head | 6.702 / 6.460 | 6.836 / 6.576 | -2.00% |
| 320 | pedestrian | 14.338 / 13.984 | 15.894 / 15.725 | -10.85% |
| 640 | frontal | 24.858 / 24.287 | 25.225 / 25.081 | -1.48% |
| 640 | profile | 24.018 / 24.018 | 24.371 / 24.202 | -1.47% |
| 640 | raised head | 26.622 / 26.014 | 24.476 / 24.287 | +8.06% |
| 640 | pedestrian | 39.020 / 38.807 | 37.826 / 37.491 | +3.06% |

Two paired rounds produced opposite noise-level aggregate results for the same fixed 12-case workload:

| Timed iterations per case | Before aggregate | After aggregate | Delta |
|---:|---:|---:|---:|
| 50 | 147.753 ms | 150.718 ms | -2.01% |
| 100 | 160.613 ms | 159.505 ms | +0.69% |

## Conclusion

Correctness is unchanged. The redundant work is removed, but this machine did not show a stable, measurable end-to-end speedup; the observed difference is within run-to-run noise. Treat this change as a safe code-path cleanup, not as a demonstrated performance win.
