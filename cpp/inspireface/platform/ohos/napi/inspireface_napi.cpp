#include <napi/native_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "c_api/inspireface.h"
#include "ohos_napi_contract.h"

namespace {

constexpr napi_type_tag kSessionTypeTag = {0x8e68a46fd3b9460dULL, 0xa37adf709e45b06bULL};
constexpr napi_type_tag kImageStreamTypeTag = {0x20e8d7e8040f43b1ULL, 0xa2d11c912ea59bbfULL};

struct SessionState {
    std::mutex mutex;
    HFSession handle = nullptr;

    ~SessionState() {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle) {
            HFReleaseInspireFaceSession(handle);
            handle = nullptr;
        }
    }
};

struct ImageStreamState {
    std::mutex mutex;
    std::vector<uint8_t> bytes;
    HFImageStream handle = nullptr;

    ~ImageStreamState() {
        std::lock_guard<std::mutex> lock(mutex);
        if (handle) {
            HFReleaseImageStream(handle);
            handle = nullptr;
        }
    }
};

struct SnapshotGuard {
    HFFaceResultSnapshot handle = nullptr;

    ~SnapshotGuard() {
        if (handle) {
            HFReleaseFaceResultSnapshot(handle);
        }
    }
};

bool CheckNapi(napi_env env, napi_status status, const char* operation) {
    if (status == napi_ok) {
        return true;
    }
    const napi_extended_error_info* info = nullptr;
    napi_get_last_error_info(env, &info);
    std::string message(operation ? operation : "Node-API operation");
    message += " failed";
    if (info && info->error_message) {
        message += ": ";
        message += info->error_message;
    }
    napi_throw_error(env, "ERR_INSPIREFACE_NAPI", message.c_str());
    return false;
}

napi_value ThrowTypeError(napi_env env, const char* message) {
    napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", message);
    return nullptr;
}

napi_value ThrowRangeError(napi_env env, const char* message) {
    napi_throw_range_error(env, "ERR_INSPIREFACE_RANGE", message);
    return nullptr;
}

std::string GetSdkErrorMessage(HResult result) {
    HInt32 required = 0;
    if (HFGetErrorMessage(result, nullptr, 0, &required) != HSUCCEED || required <= 0 || required > 4096) {
        return "InspireFace operation failed";
    }
    std::vector<char> buffer(static_cast<size_t>(required), '\0');
    if (HFGetErrorMessage(result, buffer.data(), required, &required) != HSUCCEED) {
        return "InspireFace operation failed";
    }
    return std::string(buffer.data());
}

napi_value ThrowSdkError(napi_env env, HResult result, const char* operation) {
    std::string message(operation ? operation : "InspireFace operation");
    message += " failed: ";
    message += GetSdkErrorMessage(result);
    const std::string code = std::to_string(static_cast<int64_t>(result));
    napi_throw_error(env, code.c_str(), message.c_str());
    return nullptr;
}

template <typename Callable>
napi_value GuardCallback(napi_env env, Callable&& callback) noexcept {
    try {
        return callback();
    } catch (const std::bad_alloc&) {
        napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate native bridge memory");
    } catch (const std::exception& error) {
        napi_throw_error(env, "ERR_INSPIREFACE_NATIVE", error.what());
    } catch (...) {
        napi_throw_error(env, "ERR_INSPIREFACE_NATIVE", "Unknown native bridge failure");
    }
    return nullptr;
}

napi_value Undefined(napi_env env) {
    napi_value value = nullptr;
    return CheckNapi(env, napi_get_undefined(env, &value), "get undefined") ? value : nullptr;
}

bool ReadArguments(napi_env env, napi_callback_info info, size_t required, napi_value* arguments) {
    size_t count = required;
    if (!CheckNapi(env, napi_get_cb_info(env, info, &count, arguments, nullptr, nullptr), "read arguments")) {
        return false;
    }
    if (count != required) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Unexpected number of arguments");
        return false;
    }
    return true;
}

bool ReadString(napi_env env, napi_value value, std::string* result) {
    if (!result) {
        return false;
    }
    size_t length = 0;
    if (!CheckNapi(env, napi_get_value_string_utf8(env, value, nullptr, 0, &length), "read string length")) {
        return false;
    }
    std::vector<char> buffer(length + 1, '\0');
    size_t copied = 0;
    if (!CheckNapi(env, napi_get_value_string_utf8(env, value, buffer.data(), buffer.size(), &copied), "read string")) {
        return false;
    }
    result->assign(buffer.data(), copied);
    return true;
}

bool GetOptionalInt32(napi_env env, napi_value object, const char* name, int32_t* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check numeric option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read numeric option") ||
        !CheckNapi(env, napi_get_value_int32(env, property, value), "convert numeric option")) {
        return false;
    }
    *present = true;
    return true;
}

bool GetOptionalInt64(napi_env env, napi_value object, const char* name, int64_t* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check integer option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read integer option") ||
        !CheckNapi(env, napi_get_value_int64(env, property, value), "convert integer option")) {
        return false;
    }
    *present = true;
    return true;
}

bool GetOptionalDouble(napi_env env, napi_value object, const char* name, double* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check floating-point option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read floating-point option") ||
        !CheckNapi(env, napi_get_value_double(env, property, value), "convert floating-point option")) {
        return false;
    }
    *present = true;
    return true;
}

bool GetOptionalBool(napi_env env, napi_value object, const char* name, bool* value, bool* present) {
    if (!value || !present) {
        return false;
    }
    *present = false;
    bool has_property = false;
    if (!CheckNapi(env, napi_has_named_property(env, object, name, &has_property), "check boolean option")) {
        return false;
    }
    if (!has_property) {
        return true;
    }
    napi_value property = nullptr;
    if (!CheckNapi(env, napi_get_named_property(env, object, name, &property), "read boolean option") ||
        !CheckNapi(env, napi_get_value_bool(env, property, value), "convert boolean option")) {
        return false;
    }
    *present = true;
    return true;
}

template <typename State>
void FinalizeState(napi_env, void* data, void*) {
    delete static_cast<State*>(data);
}

template <typename State>
napi_value WrapState(napi_env env, State* state, const napi_type_tag& type_tag) {
    if (!state) {
        napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate native handle");
        return nullptr;
    }
    napi_value object = nullptr;
    if (!CheckNapi(env, napi_create_object(env, &object), "create native handle object") ||
        !CheckNapi(env, napi_type_tag_object(env, object, &type_tag), "tag native handle object") ||
        !CheckNapi(env, napi_wrap(env, object, state, FinalizeState<State>, nullptr, nullptr), "wrap native handle")) {
        delete state;
        return nullptr;
    }
    return object;
}

template <typename State>
State* UnwrapState(napi_env env, napi_value object, const napi_type_tag& type_tag, const char* expected_type) {
    bool tagged = false;
    if (!CheckNapi(env, napi_check_object_type_tag(env, object, &type_tag, &tagged), "check native handle type")) {
        return nullptr;
    }
    if (!tagged) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_HANDLE", expected_type);
        return nullptr;
    }
    void* state = nullptr;
    if (!CheckNapi(env, napi_unwrap(env, object, &state), "unwrap native handle") || !state) {
        return nullptr;
    }
    return static_cast<State*>(state);
}

bool ReadUint8View(napi_env env, napi_value value, const uint8_t** data, size_t* length) {
    napi_typedarray_type type = napi_uint8_array;
    void* raw_data = nullptr;
    napi_value array_buffer = nullptr;
    size_t byte_offset = 0;
    if (!CheckNapi(env, napi_get_typedarray_info(env, value, &type, length, &raw_data, &array_buffer, &byte_offset),
                   "read Uint8Array")) {
        return false;
    }
    if (type != napi_uint8_array && type != napi_uint8_clamped_array) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Expected Uint8Array or Uint8ClampedArray");
        return false;
    }
    if (*length > 0 && !raw_data) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Typed array has no backing storage");
        return false;
    }
    *data = static_cast<const uint8_t*>(raw_data);
    return true;
}

bool ReadFloat32View(napi_env env, napi_value value, float** data, size_t* length) {
    napi_typedarray_type type = napi_float32_array;
    void* raw_data = nullptr;
    napi_value array_buffer = nullptr;
    size_t byte_offset = 0;
    if (!CheckNapi(env, napi_get_typedarray_info(env, value, &type, length, &raw_data, &array_buffer, &byte_offset),
                   "read Float32Array")) {
        return false;
    }
    if (type != napi_float32_array) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Expected Float32Array");
        return false;
    }
    if (*length > 0 && !raw_data) {
        napi_throw_type_error(env, "ERR_INSPIREFACE_ARGUMENT", "Typed array has no backing storage");
        return false;
    }
    *data = static_cast<float*>(raw_data);
    return true;
}

napi_value CreateTypedArray(napi_env env, napi_typedarray_type type, const void* source, size_t element_count,
                            size_t element_size) {
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
        return ThrowRangeError(env, "Typed array size overflow");
    }
    const size_t byte_count = element_count * element_size;
    void* destination = nullptr;
    napi_value array_buffer = nullptr;
    napi_value typed_array = nullptr;
    if (!CheckNapi(env, napi_create_arraybuffer(env, byte_count, &destination, &array_buffer), "create ArrayBuffer")) {
        return nullptr;
    }
    if (byte_count > 0) {
        if (!source || !destination) {
            return ThrowTypeError(env, "Typed array data is unavailable");
        }
        std::memcpy(destination, source, byte_count);
    }
    if (!CheckNapi(env, napi_create_typedarray(env, type, element_count, array_buffer, 0, &typed_array),
                   "create typed array")) {
        return nullptr;
    }
    return typed_array;
}

bool SetInt32(napi_env env, napi_value object, const char* name, int32_t value) {
    napi_value number = nullptr;
    return CheckNapi(env, napi_create_int32(env, value, &number), "create integer") &&
           CheckNapi(env, napi_set_named_property(env, object, name, number), "set integer property");
}

bool SetDouble(napi_env env, napi_value object, const char* name, double value) {
    napi_value number = nullptr;
    return CheckNapi(env, napi_create_double(env, value, &number), "create number") &&
           CheckNapi(env, napi_set_named_property(env, object, name, number), "set number property");
}

napi_value Launch(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        std::string resource_path;
        if (!ReadString(env, arguments[0], &resource_path) || resource_path.empty()) {
            return resource_path.empty() ? ThrowTypeError(env, "resourcePath must not be empty") : nullptr;
        }
        const HResult result = HFLaunchInspireFace(resource_path.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "launch");
    });
}

napi_value Reload(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        std::string resource_path;
        if (!ReadString(env, arguments[0], &resource_path) || resource_path.empty()) {
            return resource_path.empty() ? ThrowTypeError(env, "resourcePath must not be empty") : nullptr;
        }
        const HResult result = HFReloadInspireFace(resource_path.c_str());
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "reload");
    });
}

napi_value Terminate(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        const HResult result = HFTerminateInspireFace();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "terminate");
    });
}

napi_value IsLaunched(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HInt32 launched = 0;
        const HResult result = HFQueryInspireFaceLaunchStatus(&launched);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query launch status");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_get_boolean(env, launched != 0, &value), "create launch status") ? value : nullptr;
    });
}

napi_value GetVersion(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFInspireFaceVersion version{};
        const HResult result = HFQueryInspireFaceVersion(&version);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query version");
        }
        napi_value object = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &object), "create version object") ||
            !SetInt32(env, object, "major", version.major) || !SetInt32(env, object, "minor", version.minor) ||
            !SetInt32(env, object, "patch", version.patch)) {
            return nullptr;
        }
        return object;
    });
}

napi_value GetCapiLevel(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFUInt32 api_level = 0;
        const HFStatus result = HFQueryCAPILevel(&api_level);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query C API level");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_uint32(env, api_level, &value), "create C API level") ? value : nullptr;
    });
}

napi_value CreateSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }

        HFSessionConfigV2 config{};
        config.structSize = sizeof(config);
        config.structVersion = HF_SESSION_CONFIG_V2_VERSION;
        config.featureMask = HF_ENABLE_NONE;
        config.detectMode = HF_DETECT_MODE_ALWAYS_DETECT;
        config.maxDetectFaceNum = 1;
        config.detectPixelLevel = -1;
        config.trackByDetectModeFPS = -1;

        bool present = false;
        int64_t feature_mask = 0;
        if (!GetOptionalInt64(env, arguments[0], "featureMask", &feature_mask, &present)) {
            return nullptr;
        }
        if (present) {
            if (feature_mask < 0) {
                return ThrowRangeError(env, "featureMask must be non-negative");
            }
            config.featureMask = static_cast<HFUInt64>(feature_mask);
        }
        if (!GetOptionalInt32(env, arguments[0], "detectMode", &config.detectMode, &present) ||
            !GetOptionalInt32(env, arguments[0], "maxFaces", &config.maxDetectFaceNum, &present) ||
            !GetOptionalInt32(env, arguments[0], "detectPixelLevel", &config.detectPixelLevel, &present) ||
            !GetOptionalInt32(env, arguments[0], "trackFps", &config.trackByDetectModeFPS, &present)) {
            return nullptr;
        }

        auto* state = new (std::nothrow) SessionState();
        if (!state) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate session wrapper");
            return nullptr;
        }
        const HFStatus result = HFCreateInspireFaceSessionV2(&config, &state->handle);
        if (result != HSUCCEED) {
            delete state;
            return ThrowSdkError(env, result, "create session");
        }
        return WrapState(env, state, kSessionTypeTag);
    });
}

napi_value ReleaseSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        SessionState* state = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return Undefined(env);
        }
        const HResult result = HFReleaseInspireFaceSession(state->handle);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "release session");
        }
        state->handle = nullptr;
        return Undefined(env);
    });
}

napi_value ConfigureSession(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) {
            return nullptr;
        }
        SessionState* state = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return ThrowTypeError(env, "Session has already been released");
        }

        bool present = false;
        int32_t integer = 0;
        double decimal = 0.0;
        bool boolean = false;
        HResult result = HSUCCEED;

#define APPLY_INT_OPTION(property, function_name)                                                        \
    do {                                                                                                \
        if (!GetOptionalInt32(env, arguments[1], property, &integer, &present)) return nullptr;         \
        if (present && (result = function_name(state->handle, integer)) != HSUCCEED)                    \
            return ThrowSdkError(env, result, "configure session: " property);                         \
    } while (false)
#define APPLY_FLOAT_OPTION(property, function_name)                                                      \
    do {                                                                                                \
        if (!GetOptionalDouble(env, arguments[1], property, &decimal, &present)) return nullptr;        \
        if (present && (result = function_name(state->handle, static_cast<HFloat>(decimal))) != HSUCCEED) \
            return ThrowSdkError(env, result, "configure session: " property);                         \
    } while (false)

        APPLY_INT_OPTION("previewSize", HFSessionSetTrackPreviewSize);
        APPLY_INT_OPTION("minimumFaceSize", HFSessionSetFilterMinimumFacePixelSize);
        APPLY_INT_OPTION("smoothCacheFrames", HFSessionSetTrackModeNumSmoothCacheFrame);
        APPLY_INT_OPTION("detectInterval", HFSessionSetTrackModeDetectInterval);
        APPLY_INT_OPTION("landmarkAugmentation", HFSessionSetLandmarkAugmentationNum);
        APPLY_FLOAT_OPTION("detectThreshold", HFSessionSetFaceDetectThreshold);
        APPLY_FLOAT_OPTION("smoothRatio", HFSessionSetTrackModeSmoothRatio);
        APPLY_FLOAT_OPTION("lightTrackThreshold", HFSessionSetLightTrackConfidenceThreshold);
        if (!GetOptionalBool(env, arguments[1], "trackLostRecovery", &boolean, &present)) {
            return nullptr;
        }
        if (present && (result = HFSessionSetTrackLostRecoveryMode(state->handle, boolean ? 1 : 0)) != HSUCCEED) {
            return ThrowSdkError(env, result, "configure session: trackLostRecovery");
        }

#undef APPLY_FLOAT_OPTION
#undef APPLY_INT_OPTION
        return Undefined(env);
    });
}

napi_value ClearTracking(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        SessionState* state = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return ThrowTypeError(env, "Session has already been released");
        }
        const HResult result = HFSessionClearTrackingFace(state->handle);
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "clear tracking faces");
    });
}

napi_value CreateImageStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        if (!ReadArguments(env, info, 5, arguments)) {
            return nullptr;
        }
        const uint8_t* source = nullptr;
        size_t source_size = 0;
        int32_t width = 0;
        int32_t height = 0;
        int32_t format_value = 0;
        int32_t rotation_value = 0;
        if (!ReadUint8View(env, arguments[0], &source, &source_size) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[1], &width), "read image width") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[2], &height), "read image height") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[3], &format_value), "read image format") ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[4], &rotation_value), "read image rotation")) {
            return nullptr;
        }
        if (rotation_value < HF_CAMERA_ROTATION_0 || rotation_value > HF_CAMERA_ROTATION_270) {
            return ThrowRangeError(env, "Unsupported image rotation");
        }

        auto* state = new (std::nothrow) ImageStreamState();
        if (!state) {
            napi_throw_error(env, "ERR_INSPIREFACE_NO_MEMORY", "Unable to allocate image stream wrapper");
            return nullptr;
        }
        const auto format = static_cast<HFImageFormat>(format_value);
        if (!inspire::ohos::CopyExactImageBytes(source, source_size, format, width, height, &state->bytes)) {
            delete state;
            return ThrowRangeError(env, "Image byte length, dimensions, or format are invalid");
        }

        HFImageData image_data{};
        image_data.data = state->bytes.data();
        image_data.width = width;
        image_data.height = height;
        image_data.format = format;
        image_data.rotation = static_cast<HFRotation>(rotation_value);
        const HResult result = HFCreateImageStream(&image_data, &state->handle);
        if (result != HSUCCEED) {
            delete state;
            return ThrowSdkError(env, result, "create image stream");
        }
        return WrapState(env, state, kImageStreamTypeTag);
    });
}

napi_value ReleaseImageStream(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        ImageStreamState* state =
          UnwrapState<ImageStreamState>(env, arguments[0], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!state) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->handle) {
            return Undefined(env);
        }
        const HResult result = HFReleaseImageStream(state->handle);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "release image stream");
        }
        state->handle = nullptr;
        std::vector<uint8_t>().swap(state->bytes);
        return Undefined(env);
    });
}

napi_value Track(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) {
            return nullptr;
        }
        SessionState* session = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        ImageStreamState* image =
          UnwrapState<ImageStreamState>(env, arguments[1], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        if (!session || !image) {
            return nullptr;
        }

        SnapshotGuard snapshot;
        {
            std::unique_lock<std::mutex> session_lock(session->mutex, std::defer_lock);
            std::unique_lock<std::mutex> image_lock(image->mutex, std::defer_lock);
            std::lock(session_lock, image_lock);
            if (!session->handle || !image->handle) {
                return ThrowTypeError(env, "Session or image stream has already been released");
            }
            const HResult result = HFExecuteFaceTrackSnapshot(session->handle, image->handle, &snapshot.handle);
            if (result != HSUCCEED) {
                return ThrowSdkError(env, result, "execute face track");
            }
        }

        HFMultipleFaceData faces{};
        HResult result = HFGetFaceResultSnapshotData(snapshot.handle, &faces);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "read face track snapshot");
        }
        if (faces.detectedNum < 0) {
            return ThrowSdkError(env, HERR_INVALID_FACE_LIST, "read face track snapshot");
        }
        if (faces.detectedNum > 0 &&
            (!faces.rects || !faces.trackIds || !faces.trackCounts || !faces.detConfidence || !faces.angles.roll ||
             !faces.angles.yaw || !faces.angles.pitch || !faces.tokens)) {
            return ThrowSdkError(env, HERR_INVALID_FACE_LIST, "read face track snapshot");
        }

        napi_value output = nullptr;
        napi_value face_array = nullptr;
        if (!CheckNapi(env, napi_create_object(env, &output), "create track result") ||
            !SetInt32(env, output, "detectedNum", faces.detectedNum) ||
            !CheckNapi(env, napi_create_array_with_length(env, static_cast<size_t>(faces.detectedNum), &face_array),
                       "create face array")) {
            return nullptr;
        }

        for (int32_t index = 0; index < faces.detectedNum; ++index) {
            if (faces.tokens[index].size < 0 || (faces.tokens[index].size > 0 && !faces.tokens[index].data)) {
                return ThrowSdkError(env, HERR_INVALID_FACE_TOKEN, "read face token snapshot");
            }
            napi_value face = nullptr;
            napi_value rect = nullptr;
            napi_value token = CreateTypedArray(env, napi_uint8_array, faces.tokens[index].data,
                                                static_cast<size_t>(faces.tokens[index].size), sizeof(uint8_t));
            if (!token || !CheckNapi(env, napi_create_object(env, &face), "create face result") ||
                !CheckNapi(env, napi_create_object(env, &rect), "create face rectangle") ||
                !SetInt32(env, rect, "x", faces.rects[index].x) || !SetInt32(env, rect, "y", faces.rects[index].y) ||
                !SetInt32(env, rect, "width", faces.rects[index].width) ||
                !SetInt32(env, rect, "height", faces.rects[index].height) ||
                !CheckNapi(env, napi_set_named_property(env, face, "rect", rect), "set face rectangle") ||
                !SetInt32(env, face, "trackId", faces.trackIds[index]) ||
                !SetInt32(env, face, "trackCount", faces.trackCounts[index]) ||
                !SetDouble(env, face, "confidence", faces.detConfidence[index]) ||
                !SetDouble(env, face, "roll", faces.angles.roll[index]) ||
                !SetDouble(env, face, "yaw", faces.angles.yaw[index]) ||
                !SetDouble(env, face, "pitch", faces.angles.pitch[index]) ||
                !CheckNapi(env, napi_set_named_property(env, face, "token", token), "set face token") ||
                !CheckNapi(env, napi_set_element(env, face_array, static_cast<uint32_t>(index), face), "set face result")) {
                return nullptr;
            }
        }
        if (!CheckNapi(env, napi_set_named_property(env, output, "faces", face_array), "set face array")) {
            return nullptr;
        }
        return output;
    });
}

napi_value GetLandmarks(napi_env env, napi_callback_info info, bool dense) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        const uint8_t* token_data = nullptr;
        size_t token_size = 0;
        if (!ReadUint8View(env, arguments[0], &token_data, &token_size) ||
            token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())
                     ? ThrowRangeError(env, "Face token is too large")
                     : nullptr;
        }
        HFFaceBasicToken token{static_cast<HInt32>(token_size), const_cast<uint8_t*>(token_data)};
        HInt32 count = dense ? 0 : 5;
        HResult result = dense ? HFGetNumOfFaceDenseLandmark(&count) : HSUCCEED;
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query landmark count");
        }
        std::vector<HPoint2f> points(static_cast<size_t>(count));
        result = dense ? HFGetFaceDenseLandmarkFromFaceToken(token, points.data(), count)
                       : HFGetFaceFiveKeyPointsFromFaceToken(token, points.data(), count);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, dense ? "get dense landmarks" : "get five key points");
        }
        std::vector<float> flattened(static_cast<size_t>(count) * 2);
        for (int32_t index = 0; index < count; ++index) {
            flattened[static_cast<size_t>(index) * 2] = points[static_cast<size_t>(index)].x;
            flattened[static_cast<size_t>(index) * 2 + 1] = points[static_cast<size_t>(index)].y;
        }
        return CreateTypedArray(env, napi_float32_array, flattened.data(), flattened.size(), sizeof(float));
    });
}

napi_value GetDenseLandmarks(napi_env env, napi_callback_info info) {
    return GetLandmarks(env, info, true);
}

napi_value GetFiveKeyPoints(napi_env env, napi_callback_info info) {
    return GetLandmarks(env, info, false);
}

napi_value ExtractFeature(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[3] = {nullptr, nullptr, nullptr};
        if (!ReadArguments(env, info, 3, arguments)) {
            return nullptr;
        }
        SessionState* session = UnwrapState<SessionState>(env, arguments[0], kSessionTypeTag, "Expected InspireFace session handle");
        ImageStreamState* image =
          UnwrapState<ImageStreamState>(env, arguments[1], kImageStreamTypeTag, "Expected InspireFace image stream handle");
        const uint8_t* token_data = nullptr;
        size_t token_size = 0;
        if (!session || !image || !ReadUint8View(env, arguments[2], &token_data, &token_size)) {
            return nullptr;
        }
        if (token_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return ThrowRangeError(env, "Face token is too large");
        }
        HFFaceBasicToken token{static_cast<HInt32>(token_size), const_cast<uint8_t*>(token_data)};
        std::vector<float> copied_feature;
        {
            std::unique_lock<std::mutex> session_lock(session->mutex, std::defer_lock);
            std::unique_lock<std::mutex> image_lock(image->mutex, std::defer_lock);
            std::lock(session_lock, image_lock);
            if (!session->handle || !image->handle) {
                return ThrowTypeError(env, "Session or image stream has already been released");
            }
            HFFaceFeature feature{};
            const HResult result = HFFaceFeatureExtract(session->handle, image->handle, token, &feature);
            if (result != HSUCCEED) {
                return ThrowSdkError(env, result, "extract face feature");
            }
            if (feature.size <= 0 || !feature.data) {
                return ThrowSdkError(env, HERR_INVALID_FACE_FEATURE, "extract face feature");
            }
            copied_feature.assign(feature.data, feature.data + feature.size);
        }
        return CreateTypedArray(env, napi_float32_array, copied_feature.data(), copied_feature.size(), sizeof(float));
    });
}

napi_value GetFeatureLength(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HInt32 length = 0;
        const HResult result = HFGetFeatureLength(&length);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query feature length");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_int32(env, length, &value), "create feature length") ? value : nullptr;
    });
}

napi_value CompareFeatures(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[2] = {nullptr, nullptr};
        if (!ReadArguments(env, info, 2, arguments)) {
            return nullptr;
        }
        float* first_data = nullptr;
        float* second_data = nullptr;
        size_t first_size = 0;
        size_t second_size = 0;
        if (!ReadFloat32View(env, arguments[0], &first_data, &first_size) ||
            !ReadFloat32View(env, arguments[1], &second_data, &second_size)) {
            return nullptr;
        }
        if (first_size != second_size || first_size == 0 ||
            first_size > static_cast<size_t>(std::numeric_limits<HInt32>::max())) {
            return ThrowRangeError(env, "Face features must have the same non-zero length");
        }
        HFFaceFeature first{static_cast<HInt32>(first_size), first_data};
        HFFaceFeature second{static_cast<HInt32>(second_size), second_data};
        HFloat similarity = 0.0f;
        const HResult result = HFFaceComparison(first, second, &similarity);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "compare face features");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_double(env, similarity, &value), "create similarity") ? value : nullptr;
    });
}

napi_value GetRecommendedThreshold(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        HFloat threshold = 0.0f;
        const HResult result = HFGetRecommendedCosineThreshold(&threshold);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "query recommended threshold");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_double(env, threshold, &value), "create threshold") ? value : nullptr;
    });
}

napi_value SimilarityToPercentage(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        if (!ReadArguments(env, info, 1, arguments)) {
            return nullptr;
        }
        double similarity = 0.0;
        if (!CheckNapi(env, napi_get_value_double(env, arguments[0], &similarity), "read similarity")) {
            return nullptr;
        }
        HFloat percentage = 0.0f;
        const HResult result = HFCosineSimilarityConvertToPercentage(static_cast<HFloat>(similarity), &percentage);
        if (result != HSUCCEED) {
            return ThrowSdkError(env, result, "convert similarity");
        }
        napi_value value = nullptr;
        return CheckNapi(env, napi_create_double(env, percentage, &value), "create percentage") ? value : nullptr;
    });
}

napi_value SetLogLevel(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        napi_value arguments[1] = {nullptr};
        int32_t level = 0;
        if (!ReadArguments(env, info, 1, arguments) ||
            !CheckNapi(env, napi_get_value_int32(env, arguments[0], &level), "read log level")) {
            return nullptr;
        }
        const HResult result = HFSetLogLevel(static_cast<HFLogLevel>(level));
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "set log level");
    });
}

napi_value DisableLog(napi_env env, napi_callback_info info) {
    return GuardCallback(env, [&]() -> napi_value {
        if (!ReadArguments(env, info, 0, nullptr)) {
            return nullptr;
        }
        const HResult result = HFLogDisable();
        return result == HSUCCEED ? Undefined(env) : ThrowSdkError(env, result, "disable log");
    });
}

napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor properties[] = {
      {"launch", nullptr, Launch, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"reload", nullptr, Reload, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"terminate", nullptr, Terminate, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"isLaunched", nullptr, IsLaunched, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getVersion", nullptr, GetVersion, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getCapiLevel", nullptr, GetCapiLevel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createSession", nullptr, CreateSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"releaseSession", nullptr, ReleaseSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"configureSession", nullptr, ConfigureSession, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"clearTracking", nullptr, ClearTracking, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"createImageStream", nullptr, CreateImageStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"releaseImageStream", nullptr, ReleaseImageStream, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"track", nullptr, Track, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getDenseLandmarks", nullptr, GetDenseLandmarks, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getFiveKeyPoints", nullptr, GetFiveKeyPoints, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"extractFeature", nullptr, ExtractFeature, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getFeatureLength", nullptr, GetFeatureLength, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"compareFeatures", nullptr, CompareFeatures, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"getRecommendedThreshold", nullptr, GetRecommendedThreshold, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"similarityToPercentage", nullptr, SimilarityToPercentage, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"setLogLevel", nullptr, SetLogLevel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"disableLog", nullptr, DisableLog, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    if (!CheckNapi(env, napi_define_properties(env, exports, sizeof(properties) / sizeof(properties[0]), properties),
                   "define InspireFace exports")) {
        return nullptr;
    }
    return exports;
}

}  // namespace

NAPI_MODULE(inspireface_napi, Init)
