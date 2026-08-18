"""System, native-resource, and ImageStream cases mirroring C++ API tests."""

import ctypes
import gc
import os
import subprocess
import sys
import unittest
import weakref

import numpy as np

import inspireface as ifac
from inspireface.modules.core import HF_LOG_ERROR
from inspireface.modules.exception import InvalidInputError, ResourceError
from inspireface.param import (
    HF_CAMERA_ROTATION_0,
    HF_DETECT_MODE_ALWAYS_DETECT,
    HF_ENABLE_NONE,
    HF_STREAM_BGR,
)

from .common import (
    NativeResourceCaseMixin,
    decode_stream,
    load_image,
    managed_session,
    unreleased_session_count,
    unreleased_stream_count,
)
from .settings import MODEL_PATH


class SystemCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_launch_status_and_version(self):
        self.assertTrue(ifac.query_launch_status())
        parts = ifac.version().split(".")
        self.assertEqual(len(parts), 3)
        self.assertTrue(all(part.isdigit() for part in parts))

    def test_global_lifecycle_in_isolated_process(self):
        source = """
import os
import inspireface as ifac
model = os.environ['INSPIREFACE_SYSTEM_MODEL']
assert not ifac.query_launch_status()
assert ifac.launch(resource_path=model)
assert ifac.query_launch_status()
assert ifac.reload(resource_path=model)
assert ifac.query_launch_status()
assert ifac.terminate()
assert not ifac.query_launch_status()
"""
        environment = os.environ.copy()
        environment["INSPIREFACE_SYSTEM_MODEL"] = str(MODEL_PATH)
        completed = subprocess.run(
            [sys.executable, "-c", source],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_session_resource_registry(self):
        baseline = unreleased_session_count()
        sessions = [
            ifac.InspireFaceSession(
                HF_ENABLE_NONE,
                HF_DETECT_MODE_ALWAYS_DETECT,
            )
            for _ in range(6)
        ]
        self.assertEqual(unreleased_session_count(), baseline + len(sessions))
        for session in sessions[::2]:
            session.release()
            session._sess = None
        self.assertEqual(unreleased_session_count(), baseline + 3)
        for session in sessions[1::2]:
            session.release()
            session._sess = None
        self.assertEqual(unreleased_session_count(), baseline)

    def test_stream_resource_registry(self):
        image = load_image("bulk/pedestrian.png")
        baseline = unreleased_stream_count()
        streams = [ifac.ImageStream.load_from_cv_image(image) for _ in range(8)]
        self.assertEqual(unreleased_stream_count(), baseline + len(streams))
        for stream in streams[:3]:
            stream.release()
        self.assertEqual(unreleased_stream_count(), baseline + 5)
        for stream in streams[3:]:
            stream.release()
        self.assertEqual(unreleased_stream_count(), baseline)

    def test_session_runtime_setters(self):
        with managed_session(HF_ENABLE_NONE) as session:
            session.set_detection_confidence_threshold(0.5)
            session.set_track_preview_size(320)
            session.set_filter_minimum_face_pixel_size(16)
            session.set_track_mode_smooth_ratio(0.025)
            session.set_track_mode_num_smooth_cache_frame(5)
            session.set_track_model_detect_interval(2)
            session.set_track_lost_recovery_mode(False)
            session.set_enable_track_cost_spend(False)

    @unittest.expectedFailure
    def test_landmark_augmentation_setter_has_no_native_symbol(self):
        with managed_session(HF_ENABLE_NONE) as session:
            session.set_landmark_augmentation_num(1)

    def test_logging_controls(self):
        ifac.set_logging_level(HF_LOG_ERROR)
        ifac.disable_logging()
        ifac.set_logging_level(HF_LOG_ERROR)


class ImageStreamCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_shared_images_match_contiguous_references(self):
        for relative_path in (
            "bulk/kun.jpg",
            "bulk/yifei.jpg",
            "bulk/face_sample.png",
            "bulk/view.jpg",
        ):
            image = load_image(relative_path)
            padded = np.zeros(
                (image.shape[0], image.shape[1] * 2, image.shape[2]),
                dtype=image.dtype,
            )
            padded[:, ::2, :] = image
            view = padded[:, ::2, :]
            self.assertFalse(view.flags.c_contiguous)
            with ifac.ImageStream.load_from_cv_image(image) as reference_stream:
                reference = decode_stream(reference_stream, apply_rotation=False)
            with ifac.ImageStream.load_from_cv_image(view) as view_stream:
                actual = decode_stream(view_stream, apply_rotation=False)
                self.assertTrue(view_stream._data_owner.flags.c_contiguous)
            self.assertTrue(np.array_equal(actual, reference), relative_path)

    def test_cv_bgra_inference_and_buffer_inputs(self):
        bgr = load_image("bulk/kun_crop.jpg")
        alpha = np.full(bgr.shape[:2] + (1,), 127, dtype=np.uint8)
        bgra = np.concatenate((bgr, alpha), axis=2)
        with ifac.ImageStream.load_from_cv_image(bgra) as stream:
            self.assertEqual(stream.data_format, ifac.HF_STREAM_BGRA)
            decoded_bgra = decode_stream(stream, apply_rotation=False)
        with ifac.ImageStream.load_from_cv_image(bgr) as stream:
            decoded_bgr = decode_stream(stream, apply_rotation=False)
        self.assertLessEqual(
            int(
                np.max(
                    np.abs(
                        decoded_bgra.astype(np.int16)
                        - decoded_bgr.astype(np.int16)
                    )
                )
            ),
            1,
        )

        payload = bytearray(bgr.tobytes())
        with ifac.ImageStream.load_from_buffer(
            payload,
            bgr.shape[1],
            bgr.shape[0],
            HF_STREAM_BGR,
            HF_CAMERA_ROTATION_0,
        ) as stream:
            payload[0] ^= 0xFF
            decoded = decode_stream(stream, apply_rotation=False)
        expected = np.frombuffer(payload, dtype=np.uint8).reshape(bgr.shape).copy()
        self.assertLessEqual(
            int(np.max(np.abs(decoded.astype(np.int16) - expected.astype(np.int16)))),
            1,
        )
        payload.extend(b"\x00")

    def test_source_lifetime_and_context_release(self):
        image = load_image("bulk/yifei.jpg")
        with ifac.ImageStream.load_from_cv_image(image) as reference_stream:
            expected = decode_stream(reference_stream, apply_rotation=False)
        image_reference = weakref.ref(image)
        stream = ifac.ImageStream.load_from_cv_image(image)
        del image
        gc.collect()
        self.assertIsNotNone(image_reference())
        decoded = decode_stream(stream, apply_rotation=False)
        self.assertLessEqual(
            int(np.max(np.abs(decoded.astype(np.int16) - expected.astype(np.int16)))),
            1,
        )
        stream.release()
        stream.release()
        gc.collect()
        self.assertIsNone(image_reference())
        self.assertIsNone(stream.handle)
        with self.assertRaises(ResourceError):
            stream.write_to_file("released-stream.jpg")

    def test_invalid_stream_contract(self):
        image = load_image("bulk/kun_crop.jpg")
        invalid_operations = (
            lambda: ifac.ImageStream.load_from_cv_image(image.astype(np.float32)),
            lambda: ifac.ImageStream.load_from_cv_image(image, ifac.HF_STREAM_BGRA),
            lambda: ifac.ImageStream.load_from_buffer(
                image.tobytes()[:-1],
                image.shape[1],
                image.shape[0],
                HF_STREAM_BGR,
                HF_CAMERA_ROTATION_0,
            ),
            lambda: ifac.ImageStream.load_from_buffer(
                ctypes.POINTER(ctypes.c_uint8)(),
                image.shape[1],
                image.shape[0],
                HF_STREAM_BGR,
                HF_CAMERA_ROTATION_0,
            ),
            lambda: ifac.ImageStream.load_from_buffer(
                image.tobytes(),
                0,
                image.shape[0],
                HF_STREAM_BGR,
                HF_CAMERA_ROTATION_0,
            ),
        )
        for operation in invalid_operations:
            with self.assertRaises(InvalidInputError):
                operation()
