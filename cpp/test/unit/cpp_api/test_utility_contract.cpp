#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include <inspireface/include/inspireface/inspireface.hpp>
#include <inspireface/include/inspireface/meta.h>

#include "settings/test_settings.h"

namespace {

class SimilarityStateReset {
public:
    SimilarityStateReset() : state_(inspire::SimilarityConverter::getInstance().getState()) {}
    ~SimilarityStateReset() {
        inspire::SimilarityConverter::getInstance().updateConfigAndRecommendedThreshold(
          state_.config, state_.recommendedCosineThreshold);
    }

private:
    inspire::SimilarityConverterState state_;
};

}  // namespace

TEST_CASE("C++ SimilarityConverter validates atomically and remains monotonic", "[cpp_api][contract][similarity_converter]") {
    SimilarityStateReset reset;
    auto& converter = inspire::SimilarityConverter::getInstance();
    const inspire::SimilarityConverterConfig replacement = {0.35, 0.55, 6.0, 0.02, 0.98};
    REQUIRE(converter.updateConfigAndRecommendedThreshold(replacement, 0.42f));

    const auto state = converter.getState();
    CHECK(state.config.threshold == Approx(replacement.threshold));
    CHECK(state.config.middleScore == Approx(replacement.middleScore));
    CHECK(state.config.steepness == Approx(replacement.steepness));
    CHECK(state.config.outputMin == Approx(replacement.outputMin));
    CHECK(state.config.outputMax == Approx(replacement.outputMax));
    CHECK(state.recommendedCosineThreshold == Approx(0.42f));

    const double low = converter.convert(-1.0f);
    const double middle = converter.convert(replacement.threshold);
    const double high = converter.convert(1.0f);
    CHECK(low < middle);
    CHECK(middle < high);
    CHECK(low >= replacement.outputMin);
    CHECK(high <= replacement.outputMax);

    auto invalid = replacement;
    invalid.steepness = 0.0;
    CHECK_FALSE(converter.updateConfig(invalid));
    invalid = replacement;
    invalid.middleScore = invalid.outputMin;
    CHECK_FALSE(converter.updateConfig(invalid));
    CHECK_FALSE(converter.setRecommendedCosineThreshold(std::numeric_limits<float>::quiet_NaN()));

    const auto after_invalid = converter.getState();
    CHECK(after_invalid.config.threshold == Approx(replacement.threshold));
    CHECK(after_invalid.config.steepness == Approx(replacement.steepness));
    CHECK(after_invalid.recommendedCosineThreshold == Approx(0.42f));

    const auto config_copy = converter.getConfig();
    CHECK(config_copy.threshold == Approx(replacement.threshold));
    CHECK(converter.getRecommendedCosineThreshold() == Approx(0.42f));
    CHECK(converter.setRecommendedCosineThreshold(0.41f));
    CHECK(converter.getRecommendedCosineThreshold() == Approx(0.41f));
    CHECK(inspire::SimilarityConverter::IsConfigValid(replacement));

    inspire::SimilarityConverter local(invalid);
    CHECK(inspire::SimilarityConverter::IsConfigValid(local.getConfig()));
    const auto destroy_instance = &inspire::SimilarityConverter::destroyInstance;
    CHECK(destroy_instance != nullptr);
}

TEST_CASE("C++ version information and SpendTimer expose coherent values", "[cpp_api][contract][utility]") {
    REQUIRE(GetInspireFaceVersionMajorStr() != nullptr);
    REQUIRE(GetInspireFaceVersionMinorStr() != nullptr);
    REQUIRE(GetInspireFaceVersionPatchStr() != nullptr);
    REQUIRE(GetInspireFaceExtendedInformation() != nullptr);
    CHECK_FALSE(std::string(GetInspireFaceVersionMajorStr()).empty());
    CHECK_FALSE(std::string(GetInspireFaceVersionMinorStr()).empty());
    CHECK_FALSE(std::string(GetInspireFaceVersionPatchStr()).empty());
    CHECK_FALSE(std::string(GetInspireFaceExtendedInformation()).empty());

    const auto& sdk = inspire::GetSDKInfo();
    CHECK(sdk.GetVersionMajorStr() == GetInspireFaceVersionMajorStr());
    CHECK(sdk.GetVersionMinorStr() == GetInspireFaceVersionMinorStr());
    CHECK(sdk.GetVersionPatchStr() == GetInspireFaceVersionPatchStr());
    CHECK(sdk.GetVersionString() == sdk.GetVersionMajorStr() + "." + sdk.GetVersionMinorStr() + "." + sdk.GetVersionPatchStr());
    CHECK_FALSE(sdk.GetFullVersionInfo().empty());

    const uint64_t before = inspire::_now();
    inspire::SpendTimer timer("contract");
    CHECK(timer.name() == "contract");
    CHECK(timer.Count() == 0);
    CHECK(timer.Average() == 0);
    CHECK(timer.Min() == 0);
    CHECK(timer.Max() == 0);

    timer.Start();
    timer.Stop();
    const uint64_t after = inspire::_now();
    CHECK(after >= before);
    CHECK(timer.Count() == 1);
    CHECK(timer.Total() == timer.Get());
    CHECK(timer.Average() == timer.Total());
    CHECK(timer.Min() == timer.Get());
    CHECK(timer.Max() == timer.Get());
    CHECK(timer.Report().find("contract") != std::string::npos);

    std::ostringstream report;
    report << timer;
    CHECK(report.str() == timer.Report());

    timer.Reset();
    CHECK(timer.Count() == 0);
    CHECK(timer.Total() == 0);
    CHECK(timer.Min() == 0);
    CHECK(timer.Max() == 0);

    inspire::SpendTimer unnamed;
    CHECK(unnamed.name().empty());
    const auto disable_timer = &inspire::SpendTimer::Disable;
    CHECK(disable_timer != nullptr);
}

TEST_CASE("C++ LogManager level changes round-trip without emitting suppressed logs", "[cpp_api][contract][log]") {
    auto* logger = inspire::LogManager::getInstance();
    REQUIRE(logger != nullptr);
    CHECK(logger == inspire::LogManager::getInstance());
    const auto original = logger->getLogLevel();

    logger->setLogLevel(inspire::ISF_LOG_NONE);
    CHECK(logger->getLogLevel() == inspire::ISF_LOG_NONE);
#ifdef ANDROID
    logger->logAndroid(inspire::ISF_LOG_INFO, "InspireFaceTest", "suppressed");
#else
    logger->logStandard(inspire::ISF_LOG_INFO, "", "", -1, "suppressed");
#endif
    logger->setLogLevel(original);
    CHECK(logger->getLogLevel() == original);
}
