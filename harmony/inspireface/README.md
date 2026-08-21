# InspireFace for HarmonyOS

This HAR module adds an ArkTS API on top of the existing InspireFace C/C++ core. The Android JNI and Java APIs are unchanged.

The first adapter covers SDK launch/reload, session configuration, image streams, face tracking, stable face tokens, landmarks, feature extraction/comparison, version queries, and logging. Pipeline results such as liveness and attributes, plus FeatureHub management, remain available through the C/C++ SDK but are not yet wrapped for ArkTS.

## Build

Install or point to an OpenHarmony Native SDK and run from the repository root:

```bash
OHOS_NATIVE_HOME=/path/to/native-sdk/native ./command/build_harmonyos_napi.sh
```

The staged module is written to:

```text
build/inspireface-harmonyos-napi-arm64-v8a/install/HarmonyOS/har
```

Import that directory as a HAR module in DevEco Studio, or package it with the project's normal Hvigor workflow. The native module is already placed at `src/main/libs/arm64-v8a/libinspireface_napi.so`, and its declarations are under `src/main/cpp/types/libinspireface_napi`.

## Ownership and threading

- `ImageStream` copies the input bytes, so a camera buffer may be reused immediately after construction.
- `Session.close()` and `ImageStream.close()` are idempotent; call them deterministically instead of waiting for garbage collection.
- Calls using the same session are serialized in native code. Separate sessions may be used by separate ArkTS workers, but JavaScript wrapper objects must not be transferred between workers.
- The API is synchronous and does not create application task queues. Applications retain control over worker and scheduling policy.

See `examples/basic.ets` for the smallest tracking flow.
