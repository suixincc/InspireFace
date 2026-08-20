/**
 * Created by Jingyu Yan
 * @date 2024-10-01
 */

#ifndef INSPIREFACE_INTERNAL_H
#define INSPIREFACE_INTERNAL_H

#include "engine/face_session.h"
#include "inspireface.h"
#include <vector>

/**
 * @brief Struct for managing face algorithm session.
 * 
 * This struct holds the implementation of the face algorithm session.
 */
typedef struct HF_FaceAlgorithmSession {
    inspire::FaceSession impl;  ///< Implementation of the face context.
} HF_FaceAlgorithmSession;      ///< Handle for managing face context.

/**
 * @brief Struct for managing camera stream.
 * 
 * This struct holds the implementation of the camera stream.
 */
typedef struct HF_CameraStream {
    inspirecv::FrameProcess impl;  ///< Implementation of the camera stream.
    HFImageFormat format{HF_STREAM_YUV_NV21};  ///< Current source layout for dimension validation.
    // Normally ImageStream is a non-owning view. Streams created from an
    // ImageBitmap retain an internal snapshot so releasing or editing the
    // source bitmap cannot invalidate their pixels.
    std::vector<uint8_t> owned_buffer;
} HF_CameraStream;                 ///< Handle for managing camera stream.

/**
 * @brief Struct for managing image bitmap.
 * 
 * This struct holds the implementation of the image bitmap.
 */
typedef struct HF_ImageBitmap {
    inspirecv::Image impl;  ///< Implementation of the image bitmap.
} HF_ImageBitmap;           ///< Handle for managing image bitmap.

#endif  // INSPIREFACE_INTERNAL_H
