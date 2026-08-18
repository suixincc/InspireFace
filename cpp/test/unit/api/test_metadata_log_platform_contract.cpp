#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "inspireface/c_api/inspireface.h"
#include "settings/test_settings.h"

TEST_CASE("C API metadata outputs are initialized and null terminated", "[api][contract][metadata]") {
    HFInspireFaceVersion version = {-1, -1, -1};
    REQUIRE(HFQueryInspireFaceVersion(&version) == HSUCCEED);
    CHECK(version.major >= 0);
    CHECK(version.minor >= 0);
    CHECK(version.patch >= 0);
    CHECK(HFQueryInspireFaceVersion(nullptr) == HERR_INVALID_PARAM);

    HFInspireFaceExtendedInformation information;
    std::fill(std::begin(information.information), std::end(information.information), static_cast<char>(0x7f));
    REQUIRE(HFQueryInspireFaceExtendedInformation(&information) == HSUCCEED);
    CHECK(std::memchr(information.information, '\0', sizeof(information.information)) != nullptr);
    CHECK(std::strlen(information.information) > 0);
    CHECK(HFQueryInspireFaceExtendedInformation(nullptr) == HERR_INVALID_PARAM);
}

TEST_CASE("C API logging validates levels and safely truncates long messages", "[api][contract][logging]") {
    CHECK(HFSetLogLevel(static_cast<HFLogLevel>(-1)) == HERR_INVALID_PARAM);
    CHECK(HFSetLogLevel(static_cast<HFLogLevel>(99)) == HERR_INVALID_PARAM);
    REQUIRE(HFSetLogLevel(HF_LOG_ERROR) == HSUCCEED);
    CHECK(HFLogPrint(HF_LOG_INFO, "filtered message") == HSUCCEED);
    CHECK(HFLogPrint(static_cast<HFLogLevel>(99), "invalid") == HERR_INVALID_PARAM);
    CHECK(HFLogPrint(HF_LOG_ERROR, nullptr) == HERR_INVALID_PARAM);

    const std::string long_message(4096, 'x');
    CHECK(HFLogPrint(HF_LOG_ERROR, "%s", long_message.c_str()) == HSUCCEED);
    CHECK(HFLogDisable() == HSUCCEED);
    CHECK(HFLogPrint(HF_LOG_ERROR, "disabled message") == HSUCCEED);
    CHECK(HFSetLogLevel(HF_LOG_INFO) == HSUCCEED);
}

TEST_CASE("C API platform configuration reports supported or disabled behavior", "[api][contract][platform]") {
    HInt32 rga_enabled = -1;
    REQUIRE(HFQueryExpansiveHardwareRGACompileOption(&rga_enabled) == HSUCCEED);
    CHECK((rga_enabled == 0 || rga_enabled == 1));
    CHECK(HFQueryExpansiveHardwareRGACompileOption(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFSwitchLandmarkEngine(HF_LANDMARK_HYPLMV2_0_25) == HSUCCEED);
    CHECK(HFSwitchLandmarkEngine(static_cast<HFSessionLandmarkEngine>(99)) == HERR_INVALID_PARAM);

    std::array<char, 1024> original_path = {};
    REQUIRE(HFQueryExpansiveHardwareRockchipDmaHeapPath(original_path.data()) == HSUCCEED);
    REQUIRE(HFSetExpansiveHardwareRockchipDmaHeapPath("/dev/dma_heap/contract-test") == HSUCCEED);
    std::array<char, 1024> actual_path = {};
    REQUIRE(HFQueryExpansiveHardwareRockchipDmaHeapPath(actual_path.data()) == HSUCCEED);
    CHECK(std::string(actual_path.data()) == "/dev/dma_heap/contract-test");
    CHECK(HFSetExpansiveHardwareRockchipDmaHeapPath(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFQueryExpansiveHardwareRockchipDmaHeapPath(nullptr) == HERR_INVALID_PARAM);
    REQUIRE(HFSetExpansiveHardwareRockchipDmaHeapPath(original_path.data()) == HSUCCEED);

    CHECK(HFSetAppleCoreMLInferenceMode(static_cast<HFAppleCoreMLInferenceMode>(99)) == HERR_INVALID_PARAM);
    CHECK(HFSwitchImageProcessingBackend(static_cast<HFImageProcessingBackend>(99)) == HERR_INVALID_PARAM);
    CHECK(HFSetImageProcessAlignedWidth(0) == HERR_INVALID_PARAM);
    CHECK(HFSetCudaDeviceId(-1) == HERR_INVALID_PARAM);
    CHECK(HFGetCudaDeviceId(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFGetNumCudaDevices(nullptr) == HERR_INVALID_PARAM);
    CHECK(HFCheckCudaDeviceSupport(nullptr) == HERR_INVALID_PARAM);

    HInt32 value = -1;
    const HResult count_result = HFGetNumCudaDevices(&value);
    CHECK((count_result == HSUCCEED || count_result == HERR_DEVICE_CUDA_DISABLE));
    const HResult support_result = HFCheckCudaDeviceSupport(&value);
    CHECK((support_result == HSUCCEED || support_result == HERR_DEVICE_CUDA_DISABLE));
    const HResult print_result = HFPrintCudaDeviceInfo();
    CHECK((print_result == HSUCCEED || print_result == HERR_DEVICE_CUDA_DISABLE));
}
