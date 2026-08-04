#include "frame_process.h"
#include <vector>
#if defined(ISF_ENABLE_INSPIRECV_TASK_PREPROCESS)
#include <inspirecv/task/task.h>
#else
#include <MNN/ImageProcess.hpp>
#endif
#include "isf_check.h"

namespace inspirecv {

namespace frame_backend {

#if defined(ISF_ENABLE_INSPIRECV_TASK_PREPROCESS)
using Matrix = task::Matrix;
using Point = task::Point;
using Config = task::StreamTask::Config;
constexpr auto kBilinear = task::BILINEAR;
constexpr auto kZero = task::ZERO;
constexpr auto kNV21 = task::YUV_NV21;
constexpr auto kNV12 = task::YUV_NV12;
constexpr auto kRGBA = task::RGBA;
constexpr auto kRGB = task::RGB;
constexpr auto kBGR = task::BGR;
constexpr auto kBGRA = task::BGRA;
constexpr auto kI420 = task::YUV_I420;
constexpr auto kGray = task::GRAY;

bool Convert(const Config& config, const Matrix& matrix, const uint8_t* source,
             int sourceWidth, int sourceHeight, uint8_t* dest,
             int destWidth, int destHeight) {
    task::StreamTask* process = task::StreamTask::Create(config);
    if (process == nullptr) return false;
    process->SetMatrix(matrix);
    const auto status = process->Convert(source, sourceWidth, sourceHeight, 0,
                                         dest, destWidth, destHeight, 3, 0,
                                         halide_type_of<uint8_t>());
    task::StreamTask::Destroy(process);
    return status == SUCCESS;
}
#else
using Matrix = MNN::CV::Matrix;
using Point = MNN::CV::Point;
using Config = MNN::CV::ImageProcess::Config;
constexpr auto kBilinear = MNN::CV::BILINEAR;
constexpr auto kZero = MNN::CV::ZERO;
constexpr auto kNV21 = MNN::CV::YUV_NV21;
constexpr auto kNV12 = MNN::CV::YUV_NV12;
constexpr auto kRGBA = MNN::CV::RGBA;
constexpr auto kRGB = MNN::CV::RGB;
constexpr auto kBGR = MNN::CV::BGR;
constexpr auto kBGRA = MNN::CV::BGRA;
constexpr auto kI420 = MNN::CV::YUV_I420;
constexpr auto kGray = MNN::CV::GRAY;

bool Convert(const Config& config, const Matrix& matrix, const uint8_t* source,
             int sourceWidth, int sourceHeight, uint8_t* dest,
             int destWidth, int destHeight) {
    std::shared_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
    if (process == nullptr) return false;
    process->setMatrix(matrix);
    std::shared_ptr<MNN::Tensor> tensor(MNN::Tensor::create<uint8_t>(
        std::vector<int>{1, destHeight, destWidth, 3}, dest));
    return process->convert(source, sourceWidth, sourceHeight, 0, tensor.get()) ==
           MNN::ErrorCode::NO_ERROR;
}
#endif

}  // namespace frame_backend

class FrameProcess::Impl {
public:
    Impl() : buffer_(nullptr), height_(0), width_(0), preview_scale_(0), preview_size_(192), rotation_mode_(ROTATION_0) {
        SetDataFormat(NV21);
        SetDestFormat(BGR);
        config_.filterType = frame_backend::kBilinear;
        config_.wrap = frame_backend::kZero;
    }

    void SetDataFormat(DATA_FORMAT data_format) {
        if (data_format == NV21) {
            config_.sourceFormat = frame_backend::kNV21;
        }
        if (data_format == NV12) {
            config_.sourceFormat = frame_backend::kNV12;
        }
        if (data_format == RGBA) {
            config_.sourceFormat = frame_backend::kRGBA;
        }
        if (data_format == RGB) {
            config_.sourceFormat = frame_backend::kRGB;
        }
        if (data_format == BGR) {
            config_.sourceFormat = frame_backend::kBGR;
        }
        if (data_format == BGRA) {
            config_.sourceFormat = frame_backend::kBGRA;
        }
        if (data_format == I420) {
            config_.sourceFormat = frame_backend::kI420;
        }
        if (data_format == GRAY) {
            config_.sourceFormat = frame_backend::kGray;
        }
    }

    void SetDestFormat(DATA_FORMAT data_format) {
        if (data_format == NV21) {
            config_.destFormat = frame_backend::kNV21;
        }
        if (data_format == NV12) {
            config_.destFormat = frame_backend::kNV12;
        }
        if (data_format == RGBA) {
            config_.destFormat = frame_backend::kRGBA;
        }
        if (data_format == RGB) {
            config_.destFormat = frame_backend::kRGB;
        }
        if (data_format == BGR) {
            config_.destFormat = frame_backend::kBGR;
        }
        if (data_format == BGRA) {
            config_.destFormat = frame_backend::kBGRA;
        }
        if (data_format == I420) {
            config_.destFormat = frame_backend::kI420;
        }
        if (data_format == GRAY) {
            config_.destFormat = frame_backend::kGray;
        }
    }

    void UpdateTransformMatrix() {
        float srcPoints[] = {0.0f, 0.0f, 0.0f, (float)(height_ - 1), (float)(width_ - 1), 0.0f, (float)(width_ - 1), (float)(height_ - 1)};

        float dstPoints[8];
        if (rotation_mode_ == ROTATION_270) {
            float points[] = {(float)(height_ * preview_scale_ - 1),
                              0.0f,
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              (float)(width_ * preview_scale_ - 1)};
            memcpy(dstPoints, points, sizeof(points));
        } else if (rotation_mode_ == ROTATION_90) {
            float points[] = {0.0f,
                              (float)(width_ * preview_scale_ - 1),
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              0.0f};
            memcpy(dstPoints, points, sizeof(points));
        } else if (rotation_mode_ == ROTATION_180) {
            float points[] = {(float)(width_ * preview_scale_ - 1),
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              0.0f,
                              0.0f};
            memcpy(dstPoints, points, sizeof(points));
        } else {  // ROTATION_0
            float points[] = {0.0f,
                              0.0f,
                              0.0f,
                              (float)(height_ * preview_scale_ - 1),
                              (float)(width_ * preview_scale_ - 1),
                              0.0f,
                              (float)(width_ * preview_scale_ - 1),
                              (float)(height_ * preview_scale_ - 1)};
            memcpy(dstPoints, points, sizeof(points));
        }

        tr_.setPolyToPoly(reinterpret_cast<frame_backend::Point*>(dstPoints),
                          reinterpret_cast<frame_backend::Point*>(srcPoints), 4);
    }

    const uint8_t *buffer_;                 // Pointer to the data buffer.
    int height_;                            // Height of the camera stream image.
    int width_;                             // Width of the camera stream image.
    float preview_scale_;                   // Scaling factor for the preview image.
    int preview_size_;                      // Size of the preview image.
    frame_backend::Matrix tr_;              // Output-to-input transformation matrix.
    ROTATION_MODE rotation_mode_;           // Current rotation mode.
    frame_backend::Config config_;          // Image processing configuration.
};

FrameProcess FrameProcess::Create(const uint8_t *data_buffer, int height, int width, DATA_FORMAT data_format, ROTATION_MODE rotation_mode) {
    FrameProcess process;
    process.SetDataBuffer(data_buffer, height, width);
    process.SetDataFormat(data_format);
    process.SetRotationMode(rotation_mode);
    return process;
}

FrameProcess FrameProcess::Create(const inspirecv::Image &image, DATA_FORMAT data_format, ROTATION_MODE rotation_mode) {
    return Create(image.Data(), image.Height(), image.Width(), data_format, rotation_mode);
}

FrameProcess::FrameProcess() : pImpl(std::make_unique<Impl>()) {
    pImpl->UpdateTransformMatrix();
}

FrameProcess::~FrameProcess() = default;

FrameProcess::FrameProcess(const FrameProcess &other) : pImpl(std::make_unique<Impl>(*other.pImpl)) {}

FrameProcess::FrameProcess(FrameProcess &&other) noexcept = default;

FrameProcess &FrameProcess::operator=(const FrameProcess &other) {
    if (this != &other) {
        *pImpl = *other.pImpl;
    }
    return *this;
}

FrameProcess &FrameProcess::operator=(FrameProcess &&other) noexcept = default;

void FrameProcess::SetDataBuffer(const uint8_t *data_buffer, int height, int width) {
    pImpl->buffer_ = data_buffer;
    pImpl->height_ = height;
    pImpl->width_ = width;
    pImpl->preview_scale_ = pImpl->preview_size_ / static_cast<float>(std::max(height, width));
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetPreviewSize(const int size) {
    pImpl->preview_size_ = size;
    pImpl->preview_scale_ = pImpl->preview_size_ / static_cast<float>(std::max(pImpl->height_, pImpl->width_));
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetPreviewScale(const float scale) {
    pImpl->preview_scale_ = scale;
    pImpl->preview_size_ = static_cast<int>(pImpl->preview_scale_ * std::max(pImpl->height_, pImpl->width_));
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetRotationMode(ROTATION_MODE mode) {
    pImpl->rotation_mode_ = mode;
    pImpl->UpdateTransformMatrix();
}

void FrameProcess::SetDataFormat(DATA_FORMAT data_format) {
    pImpl->SetDataFormat(data_format);
}

void FrameProcess::SetDestFormat(DATA_FORMAT data_format) {
    pImpl->SetDestFormat(data_format);
}

float FrameProcess::GetPreviewScale() {
    return pImpl->preview_scale_;
}

inspirecv::TransformMatrix FrameProcess::GetAffineMatrix() const {
    auto affine_matrix = inspirecv::TransformMatrix::Create();
    affine_matrix[0] = pImpl->tr_[0];
    affine_matrix[1] = pImpl->tr_[1];
    affine_matrix[2] = pImpl->tr_[2];
    affine_matrix[3] = pImpl->tr_[3];
    affine_matrix[4] = pImpl->tr_[4];
    affine_matrix[5] = pImpl->tr_[5];
    return affine_matrix;
}

int FrameProcess::GetHeight() const {
    return pImpl->height_;
}

int FrameProcess::GetWidth() const {
    return pImpl->width_;
}

ROTATION_MODE FrameProcess::getRotationMode() const {
    return pImpl->rotation_mode_;
}

inspirecv::Image FrameProcess::ExecuteImageAffineProcessing(inspirecv::TransformMatrix &affine_matrix, const int width_out,
                                                            const int height_out) const {
    int sw = pImpl->width_;
    int sh = pImpl->height_;
    int rot_sw = sw;
    int rot_sh = sh;
    frame_backend::Matrix tr;
    std::vector<float> tr_cv({1, 0, 0, 0, 1, 0, 0, 0, 1});
    memcpy(tr_cv.data(), affine_matrix.Squeeze().data(), sizeof(float) * 6);
    tr.set9(tr_cv.data());
    frame_backend::Matrix tr_inv;
    tr.invert(&tr_inv);
    auto img_out = inspirecv::Image::Create(width_out, height_out, 3);
    const bool converted = frame_backend::Convert(pImpl->config_, tr_inv, pImpl->buffer_, sw, sh,
                                                   const_cast<uint8_t*>(img_out.Data()), width_out, height_out);
    INSPIREFACE_CHECK_MSG(converted, "Image preprocessing failed");
    return img_out;
}

inspirecv::Image FrameProcess::ExecutePreviewImageProcessing(bool with_rotation) {
    return ExecuteImageScaleProcessing(pImpl->preview_scale_, with_rotation);
}

inspirecv::Image FrameProcess::ExecuteImageScaleProcessing(const float scale, bool with_rotation) {
    int sw = pImpl->width_;
    int sh = pImpl->height_;
    int rot_sw = sw;
    int rot_sh = sh;
    if (pImpl->rotation_mode_ == ROTATION_270 && with_rotation) {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          (float)(pImpl->height_ * scale - 1), 0.0f, 0.0f, 0.0f, (float)(pImpl->height_ * scale - 1), (float)(pImpl->width_ * scale - 1), 0.0f,
          (float)(pImpl->width_ * scale - 1)};

        pImpl->tr_.setPolyToPoly(reinterpret_cast<frame_backend::Point*>(dstPoints),
                                reinterpret_cast<frame_backend::Point*>(srcPoints), 4);
        int scaled_height = static_cast<int>(pImpl->width_ * scale);
        int scaled_width = static_cast<int>(pImpl->height_ * scale);
        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        const bool converted = frame_backend::Convert(pImpl->config_, pImpl->tr_, pImpl->buffer_, sw, sh,
                                                       const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height);
        INSPIREFACE_CHECK_MSG(converted, "Image preprocessing failed");
        return img_out;
    } else if (pImpl->rotation_mode_ == ROTATION_90 && with_rotation) {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          0.0f,
          (float)(pImpl->width_ * scale - 1),
          (float)(pImpl->height_ * scale - 1),
          (float)(pImpl->width_ * scale - 1),
          0.0f,
          0.0f,
          (float)(pImpl->height_ * scale - 1),
          0.0f,
        };
        pImpl->tr_.setPolyToPoly(reinterpret_cast<frame_backend::Point*>(dstPoints),
                                reinterpret_cast<frame_backend::Point*>(srcPoints), 4);
        int scaled_height = static_cast<int>(pImpl->width_ * scale);
        int scaled_width = static_cast<int>(pImpl->height_ * scale);
        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        const bool converted = frame_backend::Convert(pImpl->config_, pImpl->tr_, pImpl->buffer_, sw, sh,
                                                       const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height);
        INSPIREFACE_CHECK_MSG(converted, "Image preprocessing failed");
        return img_out;
    } else if (pImpl->rotation_mode_ == ROTATION_180 && with_rotation) {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          (float)(pImpl->width_ * scale - 1),
          (float)(pImpl->height_ * scale - 1),
          (float)(pImpl->width_ * scale - 1),
          0.0f,
          0.0f,
          (float)(pImpl->height_ * scale - 1),
          0.0f,
          0.0f,
        };
        pImpl->tr_.setPolyToPoly(reinterpret_cast<frame_backend::Point*>(dstPoints),
                                reinterpret_cast<frame_backend::Point*>(srcPoints), 4);
        int scaled_height = static_cast<int>(pImpl->height_ * scale);
        int scaled_width = static_cast<int>(pImpl->width_ * scale);
        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        const bool converted = frame_backend::Convert(pImpl->config_, pImpl->tr_, pImpl->buffer_, sw, sh,
                                                       const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height);
        INSPIREFACE_CHECK_MSG(converted, "Image preprocessing failed");
        return img_out;
    } else {
        float srcPoints[] = {
          0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1),
        };
        float dstPoints[] = {
          0.0f,
          0.0f,
          0.0f,
          (float)(pImpl->height_ * scale - 1),
          (float)(pImpl->width_ * scale - 1),
          0.0f,
          (float)(pImpl->width_ * scale - 1),
          (float)(pImpl->height_ * scale - 1),
        };
        pImpl->tr_.setPolyToPoly(reinterpret_cast<frame_backend::Point*>(dstPoints),
                                reinterpret_cast<frame_backend::Point*>(srcPoints), 4);
        int scaled_height = static_cast<int>(pImpl->height_ * scale);
        int scaled_width = static_cast<int>(pImpl->width_ * scale);

        inspirecv::Image img_out(scaled_width, scaled_height, 3);
        const bool converted = frame_backend::Convert(pImpl->config_, pImpl->tr_, pImpl->buffer_, sw, sh,
                                                       const_cast<uint8_t*>(img_out.Data()), scaled_width, scaled_height);
        INSPIREFACE_CHECK_MSG(converted, "Image preprocessing failed");
        return img_out;
    }
}

inspirecv::TransformMatrix FrameProcess::GetRotationModeAffineMatrix() const {
    float srcPoints[] = {0.0f, 0.0f, 0.0f, (float)(pImpl->height_ - 1), (float)(pImpl->width_ - 1), 0.0f, (float)(pImpl->width_ - 1), (float)(pImpl->height_ - 1)};
    float dstPoints[8];
    
    if (pImpl->rotation_mode_ == ROTATION_270) {
        float points[] = {(float)(pImpl->height_ - 1),
                         0.0f,
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         (float)(pImpl->width_ - 1)};
        memcpy(dstPoints, points, sizeof(points));
    } else if (pImpl->rotation_mode_ == ROTATION_90) {
        float points[] = {0.0f,
                         (float)(pImpl->width_ - 1),
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         0.0f};
        memcpy(dstPoints, points, sizeof(points));
    } else if (pImpl->rotation_mode_ == ROTATION_180) {
        float points[] = {(float)(pImpl->width_ - 1),
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         0.0f,
                         0.0f};
        memcpy(dstPoints, points, sizeof(points));
    } else {  // ROTATION_0
        float points[] = {0.0f,
                         0.0f,
                         0.0f,
                         (float)(pImpl->height_ - 1),
                         (float)(pImpl->width_ - 1),
                         0.0f,
                         (float)(pImpl->width_ - 1),
                         (float)(pImpl->height_ - 1)};
        memcpy(dstPoints, points, sizeof(points));
    }

    frame_backend::Matrix tr;
    tr.setPolyToPoly(reinterpret_cast<frame_backend::Point*>(dstPoints),
                     reinterpret_cast<frame_backend::Point*>(srcPoints), 4);
    
    auto affine_matrix = inspirecv::TransformMatrix::Create();
    affine_matrix[0] = tr[0];
    affine_matrix[1] = tr[1];
    affine_matrix[2] = tr[2];
    affine_matrix[3] = tr[3];
    affine_matrix[4] = tr[4];
    affine_matrix[5] = tr[5];
    
    return affine_matrix;
}

}  // namespace inspirecv
