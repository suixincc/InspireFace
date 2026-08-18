"""Similarity conversion and Python exception-contract cases."""

import unittest

import numpy as np

import inspireface as ifac
from inspireface.modules.exception import InvalidInputError
from inspireface.param import HF_ENABLE_NONE

from .common import NativeResourceCaseMixin, load_image, managed_session


class SimilarityConverterCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_cpp_similarity_converter_vectors(self):
        original = ifac.get_similarity_converter_config()
        cases = (
            (
                {
                    "threshold": 0.42,
                    "middleScore": 0.6,
                    "steepness": 8.0,
                    "outputMin": 0.01,
                    "outputMax": 1.0,
                },
                (-0.80, -0.20, 0.02, 0.10, 0.25, 0.30, 0.48, 0.70, 0.80, 0.90, 1.00),
                (0.0101, 0.0201, 0.0661, 0.1113, 0.2819, 0.3673, 0.7074, 0.9334, 0.9689, 0.9858, 0.9936),
            ),
            (
                {
                    "threshold": 0.32,
                    "middleScore": 0.6,
                    "steepness": 10.0,
                    "outputMin": 0.02,
                    "outputMax": 1.0,
                },
                (-0.80, -0.20, 0.02, 0.10, 0.25, 0.32, 0.50, 0.70, 0.80, 0.90, 1.00),
                (0.0200, 0.0278, 0.0860, 0.1557, 0.4302, 0.6000, 0.8997, 0.9851, 0.9945, 0.9980, 0.9992),
            ),
        )
        try:
            for configuration, points, expected_scores in cases:
                ifac.set_similarity_converter_config(configuration)
                current = ifac.get_similarity_converter_config()
                for key, value in configuration.items():
                    self.assertAlmostEqual(current[key], value, delta=1e-6)
                for point, expected in zip(points, expected_scores):
                    actual = ifac.cosine_similarity_convert_to_percentage(point)
                    self.assertAlmostEqual(actual, expected, delta=0.01)
        finally:
            ifac.set_similarity_converter_config(original)


class ErrorContractCase(NativeResourceCaseMixin, unittest.TestCase):
    def test_feature_dtype_validation(self):
        with self.assertRaises(InvalidInputError):
            ifac.feature_comparison(
                np.zeros(512, dtype=np.float64),
                np.zeros(512, dtype=np.float32),
            )
        with self.assertRaises(InvalidInputError):
            ifac.FaceIdentity(np.zeros(512, dtype=np.float64), 1)

    def test_session_rejects_invalid_image_and_pipeline_parameter(self):
        with managed_session(HF_ENABLE_NONE) as session:
            with self.assertRaises(InvalidInputError):
                session.face_detection(object())
            image = load_image("bulk/kun.jpg")
            faces = session.face_detection(image)
            self.assertGreater(len(faces), 0)
            with self.assertRaises(InvalidInputError):
                session.face_pipeline(image, faces, object())

    def test_cv_image_shape_validation(self):
        invalid_images = (
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 2), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
            np.zeros((0, 10, 3), dtype=np.uint8),
        )
        for image in invalid_images:
            with self.assertRaises(InvalidInputError):
                ifac.ImageStream.load_from_cv_image(image)
