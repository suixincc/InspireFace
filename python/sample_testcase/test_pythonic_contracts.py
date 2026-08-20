"""Compatibility gates for the Python-facing package and ctypes boundary."""

import ctypes
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import inspireface as ifac
from inspireface.modules import inspireface as api_module
from inspireface.modules.core import native
from inspireface.modules.core._library_path import get_lib_path, platform_library_spec
from inspireface.modules.exception import (
    ProcessingError,
    SystemNotReadyError,
    handle_c_api_errors,
)


class PublicPackageContractCase(unittest.TestCase):
    def test_star_export_manifest_is_explicit_and_resolvable(self):
        self.assertIsInstance(ifac.__all__, tuple)
        self.assertEqual(len(ifac.__all__), len(set(ifac.__all__)))
        self.assertNotIn("core", ifac.__all__)
        self.assertNotIn("exception", ifac.__all__)
        self.assertNotIn("modules", ifac.__all__)
        self.assertNotIn("utils", ifac.__all__)
        for name in ifac.__all__:
            self.assertTrue(hasattr(ifac, name), name)

        # Accidental historical module exports remain addressable for callers
        # that used them, but new star imports do not receive them.
        for name in (
            "core",
            "exception",
            "herror",
            "inspireface",
            "modules",
            "param",
            "utils",
        ):
            self.assertTrue(hasattr(ifac, name), name)
            self.assertNotIn(name, ifac.__all__)

    def test_python_and_native_versions_have_separate_contracts(self):
        version_file = Path(ifac.__file__).resolve().parent.parent / "version.txt"
        self.assertEqual(ifac.__version__, version_file.read_text(encoding="utf-8").strip())
        self.assertEqual(ifac.__native_version__, ifac.version())
        self.assertIs(ifac.native_version, ifac.version)

    def test_face_identity_repr_is_bounded(self):
        identity = ifac.FaceIdentity(np.arange(512, dtype=np.float32), 42)
        representation = repr(identity)
        self.assertIn("id=42", representation)
        self.assertIn("shape=(512,)", representation)
        self.assertLess(len(representation), 120)
        self.assertEqual(identity.custom_id, identity.id)


class ExceptionBoundaryContractCase(unittest.TestCase):
    def test_programming_errors_keep_their_original_type(self):
        @handle_c_api_errors("test operation")
        def fail():
            raise TypeError("programmer error")

        with self.assertRaisesRegex(TypeError, "programmer error"):
            fail()

    def test_ctypes_errors_are_translated_with_flat_context(self):
        @handle_c_api_errors("test operation")
        def fail():
            raise ctypes.ArgumentError("bad native argument")

        with self.assertRaises(ProcessingError) as raised:
            fail()
        self.assertEqual(
            raised.exception.context,
            {"original_exception": "ArgumentError"},
        )
        self.assertIsInstance(raised.exception.__cause__, ctypes.ArgumentError)

    def test_explicit_session_lifecycle_does_not_auto_launch(self):
        with patch.object(api_module, "query_launch_status", return_value=False), patch.object(
            api_module,
            "launch",
        ) as launch:
            with self.assertRaises(SystemNotReadyError):
                ifac.InspireFaceSession(ifac.HF_ENABLE_NONE, auto_launch=False)
        launch.assert_not_called()


class NativePlatformContractCase(unittest.TestCase):
    def test_generated_module_keeps_loader_and_native_symbol_compatibility(self):
        self.assertTrue(hasattr(native, "LibraryLoader"))
        self.assertTrue(hasattr(native, "add_library_search_dirs"))
        self.assertTrue(hasattr(native, "HFCreateInspireFaceSession"))
        self.assertTrue(hasattr(native, "HFFaceBasicToken"))

    def test_supported_platform_mappings(self):
        self.assertEqual(
            platform_library_spec("Darwin", "arm64"),
            ("darwin", "arm64", "libInspireFace.dylib"),
        )
        self.assertEqual(
            platform_library_spec("Linux", "x86_64"),
            ("linux", "x64", "libInspireFace.so"),
        )
        self.assertEqual(
            platform_library_spec("Windows", "AMD64"),
            ("windows", "x64", "libInspireFace.dll"),
        )

    def test_unsupported_architecture_is_not_silently_misclassified(self):
        with self.assertRaisesRegex(RuntimeError, "armv7l"):
            platform_library_spec("Linux", "armv7l")

    def test_explicit_library_override_is_validated(self):
        with self.assertRaisesRegex(RuntimeError, "not found"):
            get_lib_path(environ={"INSPIREFACE_LIBRARY_PATH": "/missing/inspireface.so"})
