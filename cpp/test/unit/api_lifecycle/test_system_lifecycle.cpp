#include <string>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"
#include "unit/test_helper/c_api_guard.h"

using inspireface_test::ScopedSdkTermination;
using inspireface_test::UniqueSession;

namespace {

HInt32 LaunchStatus() {
    HInt32 status = -1;
    REQUIRE(HFQueryInspireFaceLaunchStatus(&status) == HSUCCEED);
    return status;
}

}  // namespace

TEST_CASE("C API rejects session creation before launch", "[api][contract][lifecycle]") {
    ScopedSdkTermination cleanup;
    REQUIRE(LaunchStatus() == HF_STATUS_DISABLE);

    HFSessionCustomParameter parameter = {0};
    HFSession handle = reinterpret_cast<HFSession>(static_cast<uintptr_t>(1));
    CHECK(HFCreateInspireFaceSession(parameter, HF_DETECT_MODE_ALWAYS_DETECT, 3, -1, -1, &handle) == HERR_ARCHIVE_NOT_LOAD);
    CHECK(handle == nullptr);
}

TEST_CASE("C API launch and terminate are observable and idempotent", "[api][contract][lifecycle]") {
    ScopedSdkTermination cleanup;
    REQUIRE(LaunchStatus() == HF_STATUS_DISABLE);

    REQUIRE(HFLaunchInspireFace(GET_RUNTIME_FULLPATH_NAME.c_str()) == HSUCCEED);
    CHECK(LaunchStatus() == HF_STATUS_ENABLE);

    CHECK(HFLaunchInspireFace(GET_RUNTIME_FULLPATH_NAME.c_str()) == HSUCCEED);
    CHECK(LaunchStatus() == HF_STATUS_ENABLE);

    REQUIRE(HFTerminateInspireFace() == HSUCCEED);
    CHECK(LaunchStatus() == HF_STATUS_DISABLE);
    CHECK(HFTerminateInspireFace() == HSUCCEED);
    CHECK(LaunchStatus() == HF_STATUS_DISABLE);
}

TEST_CASE("C API failed launch leaves the SDK disabled", "[api][contract][lifecycle][boundary]") {
    ScopedSdkTermination cleanup;
    const std::string missing_path = GET_RUNTIME_FULLPATH_NAME + ".missing";

    CHECK(HFLaunchInspireFace(missing_path.c_str()) == HERR_ARCHIVE_LOAD_FAILURE);
    CHECK(LaunchStatus() == HF_STATUS_DISABLE);
}

TEST_CASE("C API lifecycle functions reject null output and path pointers", "[api][contract][lifecycle][boundary]") {
    ScopedSdkTermination cleanup;
    CHECK(HFLaunchInspireFace(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFReloadInspireFace(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFQueryInspireFaceLaunchStatus(nullptr) == HERR_INVALID_PARAM);
    CHECK(LaunchStatus() == HF_STATUS_DISABLE);
}

TEST_CASE("C API reload can initialize and failed reload preserves the active archive", "[api][contract][lifecycle]") {
    ScopedSdkTermination cleanup;
    REQUIRE(HFReloadInspireFace(GET_RUNTIME_FULLPATH_NAME.c_str()) == HSUCCEED);
    REQUIRE(LaunchStatus() == HF_STATUS_ENABLE);

    HFSession session_handle = nullptr;
    REQUIRE(HFCreateInspireFaceSessionOptional(HF_ENABLE_NONE, HF_DETECT_MODE_ALWAYS_DETECT, 1, -1, -1, &session_handle) == HSUCCEED);
    UniqueSession session(session_handle);

    const std::string missing_path = GET_RUNTIME_FULLPATH_NAME + ".missing";
    CHECK(HFReloadInspireFace(missing_path.c_str()) == HERR_ARCHIVE_LOAD_FAILURE);
    CHECK(LaunchStatus() == HF_STATUS_ENABLE);
    CHECK(session.Reset() == HSUCCEED);
}
