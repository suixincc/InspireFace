#include <array>
#include <cmath>
#include <type_traits>

#include <inspireface/include/inspireface/inspireface.hpp>

#include "settings/test_settings.h"

TEST_CASE("C++ FrameProcess preserves configuration across copy and move", "[cpp_api][contract][frame_process]") {
    static_assert(std::is_copy_constructible<inspirecv::FrameProcess>::value, "FrameProcess must remain copy constructible");
    static_assert(std::is_move_constructible<inspirecv::FrameProcess>::value, "FrameProcess must remain move constructible");

    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto original = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    original.SetPreviewScale(0.5f);

    inspirecv::FrameProcess copied(original);
    CHECK(copied.GetWidth() == image.Width());
    CHECK(copied.GetHeight() == image.Height());
    CHECK(copied.GetPreviewScale() == Approx(0.5f));
    CHECK(copied.getRotationMode() == inspirecv::ROTATION_0);

    original.SetRotationMode(inspirecv::ROTATION_90);
    original.SetPreviewScale(0.25f);
    CHECK(copied.getRotationMode() == inspirecv::ROTATION_0);
    CHECK(copied.GetPreviewScale() == Approx(0.5f));

    inspirecv::FrameProcess moved(std::move(original));
    CHECK(moved.GetWidth() == image.Width());
    CHECK(moved.getRotationMode() == inspirecv::ROTATION_90);

    // A moved-from object must at least remain assignable.
    original = copied;
    CHECK(original.GetWidth() == image.Width());
    CHECK(original.getRotationMode() == inspirecv::ROTATION_0);

    inspirecv::FrameProcess move_assigned;
    move_assigned = std::move(moved);
    CHECK(move_assigned.GetHeight() == image.Height());
    CHECK(move_assigned.getRotationMode() == inspirecv::ROTATION_90);
}

TEST_CASE("C++ FrameProcess rotations and scales produce coherent dimensions", "[cpp_api][contract][frame_process][consistency]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/r0.jpg"));
    REQUIRE(!image.Empty());

    const std::array<inspirecv::ROTATION_MODE, 4> rotations = {
      {inspirecv::ROTATION_0, inspirecv::ROTATION_90, inspirecv::ROTATION_180, inspirecv::ROTATION_270}};
    for (const auto rotation : rotations) {
        auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, rotation);
        const auto full = process.ExecuteImageScaleProcessing(1.0f, true);
        REQUIRE(!full.Empty());
        if (rotation == inspirecv::ROTATION_90 || rotation == inspirecv::ROTATION_270) {
            CHECK(full.Width() == image.Height());
            CHECK(full.Height() == image.Width());
        } else {
            CHECK(full.Width() == image.Width());
            CHECK(full.Height() == image.Height());
        }

        const auto half = process.ExecuteImageScaleProcessing(0.5f, true);
        REQUIRE(!half.Empty());
        CHECK(std::abs(half.Width() * 2 - full.Width()) <= 1);
        CHECK(std::abs(half.Height() * 2 - full.Height()) <= 1);
    }
}

TEST_CASE("C++ FrameProcess affine output honors the requested size", "[cpp_api][contract][frame_process]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/kun.jpg"));
    REQUIRE(!image.Empty());
    auto process = inspirecv::FrameProcess::Create(image, inspirecv::BGR, inspirecv::ROTATION_0);
    auto matrix = inspirecv::TransformMatrix::Create();
    const auto output = process.ExecuteImageAffineProcessing(matrix, 112, 96);
    REQUIRE(!output.Empty());
    CHECK(output.Width() == 112);
    CHECK(output.Height() == 96);
    CHECK(output.Channels() == 3);
}

TEST_CASE("C++ FrameProcess raw buffers and mutable preprocessing settings remain usable", "[cpp_api][contract][frame_process]") {
    const auto image = inspirecv::Image::Create(GET_DATA("data/bulk/r0.jpg"));
    REQUIRE(!image.Empty());

    auto process = inspirecv::FrameProcess::Create(
      image.Data(), image.Height(), image.Width(), inspirecv::BGR, inspirecv::ROTATION_0);
    CHECK(process.GetWidth() == image.Width());
    CHECK(process.GetHeight() == image.Height());

    process.SetPreviewSize(160);
    CHECK(process.GetPreviewScale() == Approx(160.0f / std::max(image.Width(), image.Height())));
    auto preview = process.ExecutePreviewImageProcessing(false);
    REQUIRE(!preview.Empty());
    CHECK(std::max(preview.Width(), preview.Height()) <= 160);

    process.SetDataBuffer(image.Data(), image.Height(), image.Width());
    process.SetDataFormat(inspirecv::BGR);
    process.SetDestFormat(inspirecv::RGB);
    process.SetRotationMode(inspirecv::ROTATION_90);
    const auto rotated = process.ExecutePreviewImageProcessing(true);
    REQUIRE(!rotated.Empty());
    CHECK(rotated.Width() == static_cast<int>(image.Height() * process.GetPreviewScale()));
    CHECK(rotated.Height() == static_cast<int>(image.Width() * process.GetPreviewScale()));

    const auto affine = process.GetAffineMatrix();
    const auto rotation_affine = process.GetRotationModeAffineMatrix();
    for (size_t index = 0; index < 6; ++index) {
        CHECK(std::isfinite(affine[index]));
        CHECK(std::isfinite(rotation_affine[index]));
    }
}
