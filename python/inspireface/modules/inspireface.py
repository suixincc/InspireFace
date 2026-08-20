import ctypes
from contextlib import contextmanager

import numpy as np
from .core import *
from typing import Tuple, List
from dataclasses import dataclass
from loguru import logger
from .utils import ResourceManager
from .utils.resource import set_use_oss_download
from . import herror as errcode
# Exception system
from .exception import (
    check_error, validate_image_format, validate_feature_data, 
    validate_session_initialized, handle_c_api_errors,
    InspireFaceError, InvalidInputError, SystemNotReadyError, 
    ProcessingError, ResourceError, HardwareError, FeatureHubError
)

# If True, the latest model will not be verified
IGNORE_VERIFICATION_OF_THE_LATEST_MODEL = False

_INT32_MAX = (1 << 31) - 1
_PACKED_STREAM_CHANNELS = {
    HF_STREAM_RGB: 3,
    HF_STREAM_BGR: 3,
    HF_STREAM_RGBA: 4,
    HF_STREAM_BGRA: 4,
    HF_STREAM_GRAY: 1,
}
_YUV420_STREAM_FORMATS = {
    HF_STREAM_YUV_NV12,
    HF_STREAM_YUV_NV21,
    HF_STREAM_I420,
}
_VALID_STREAM_FORMATS = set(_PACKED_STREAM_CHANNELS) | _YUV420_STREAM_FORMATS
_VALID_ROTATIONS = {
    HF_CAMERA_ROTATION_0,
    HF_CAMERA_ROTATION_90,
    HF_CAMERA_ROTATION_180,
    HF_CAMERA_ROTATION_270,
}


def _normalize_int32(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise InvalidInputError(
            f"{name} must be an integer",
            errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
            **{name: value}
        )
    normalized = int(value)
    if normalized < 0 or normalized > _INT32_MAX or (positive and normalized == 0):
        qualifier = "a positive 32-bit integer" if positive else "a non-negative 32-bit integer"
        raise InvalidInputError(
            f"{name} must be {qualifier}",
            errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
            **{name: normalized}
        )
    return normalized


def _normalize_stream_format(stream_format):
    normalized = _normalize_int32(stream_format, "stream_format")
    if normalized not in _VALID_STREAM_FORMATS:
        raise InvalidInputError(
            "Unsupported image stream format",
            errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
            stream_format=normalized
        )
    return normalized


def _normalize_rotation(rotation):
    normalized = _normalize_int32(rotation, "rotation")
    if normalized not in _VALID_ROTATIONS:
        raise InvalidInputError(
            "Unsupported image rotation",
            errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
            rotation=normalized
        )
    return normalized


def _required_image_bytes(width, height, stream_format):
    if stream_format in _YUV420_STREAM_FORMATS:
        return width * (height * 3 // 2)
    return width * height * _PACKED_STREAM_CHANNELS[stream_format]


def ignore_check_latest_model(ignore: bool):
    global IGNORE_VERIFICATION_OF_THE_LATEST_MODEL
    IGNORE_VERIFICATION_OF_THE_LATEST_MODEL = ignore

def use_oss_download(use_oss: bool = True):
    """Enable OSS download instead of ModelScope (for backward compatibility)
    
    Args:
        use_oss (bool): If True, use OSS download; if False, use ModelScope (default)
    """
    set_use_oss_download(use_oss)

class ImageStream(object):
    """
    ImageStream class handles the conversion of image data from various sources into a format compatible with the InspireFace library.
    It allows loading image data from numpy arrays, buffer objects, and directly from OpenCV images.

    The stream retains its input memory until release. Contiguous mutable inputs are
    used without copying, so callers must not resize them and should avoid changing
    their contents while native processing is in progress. Non-contiguous inputs are
    copied once into contiguous storage owned by the stream.
    """

    @staticmethod
    def load_from_cv_image(image: np.ndarray, stream_format=None, rotation=HF_CAMERA_ROTATION_0):
        """
        Load image data from an OpenCV image (numpy ndarray).

        Args:
            image (np.ndarray): The image data as a numpy array.
            stream_format (int, optional): The format of the image data. If omitted,
                three-channel images use BGR and four-channel images use BGRA.
            rotation (int): The rotation angle to be applied to the image data.

        Returns:
            ImageStream: An instance of the ImageStream class initialized with the provided image data.

        Raises:
            InvalidInputError: If the image does not have 3 or 4 channels.
        """
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise InvalidInputError(
                "CV image must be a uint8 numpy array",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                input_type=type(image).__name__,
                actual_dtype=str(getattr(image, "dtype", None)),
            )
        if image.ndim == 2:
            h, w = image.shape
            c = 1
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            h, w, c = image.shape
        else:
            raise InvalidInputError(
                "CV image must have shape (H, W), (H, W, 3), or (H, W, 4)",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                actual_shape=image.shape,
            )
        if h <= 0 or w <= 0:
            raise InvalidInputError(
                "CV image width and height must be positive",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                actual_shape=image.shape,
            )
        if stream_format is None:
            stream_format = {1: HF_STREAM_GRAY, 3: HF_STREAM_BGR, 4: HF_STREAM_BGRA}[c]
        else:
            stream_format = _normalize_stream_format(stream_format)
            expected_channels = _PACKED_STREAM_CHANNELS.get(stream_format)
            if expected_channels != c:
                raise InvalidInputError(
                    "CV image channels do not match stream format",
                    errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                    channels=c,
                    stream_format=stream_format,
                    expected_channels=expected_channels
                )
        return ImageStream(image, w, h, stream_format, rotation)

    @staticmethod
    def load_from_ndarray(data: np.ndarray, width: int, height: int, stream_format: int, rotation: int):
        """
        Load image data from a numpy array specifying width and height explicitly.

        Args:
            data (np.ndarray): The raw image data.
            width (int): The width of the image.
            height (int): The height of the image.
            stream_format (int): The format of the image data.
            rotation (int): The rotation angle to be applied to the image data.

        Returns:
            ImageStream: An instance of the ImageStream class.
        """
        return ImageStream(data, width, height, stream_format, rotation)

    @staticmethod
    def load_from_buffer(data, width: int, height: int, stream_format: int, rotation: int):
        """
        Load image data from a buffer (like bytes or bytearray).

        Args:
            data: The buffer containing the image data.
            width (int): The width of the image.
            height (int): The height of the image.
            stream_format (int): The format of the image data.
            rotation (int): The rotation angle to be applied to the image data.

        Returns:
            ImageStream: An instance of the ImageStream class.
        """
        return ImageStream(data, width, height, stream_format, rotation)

    def __init__(self, data, width: int, height: int, stream_format: int, rotation: int):
        """
        Initialize the ImageStream object with provided data and configuration.

        Args:
            data: The image data (numpy array or buffer).
            width (int): The width of the image.
            height (int): The height of the image.
            stream_format (int): The format of the image data.
            rotation (int): The rotation applied to the image.

        Raises:
            InvalidInputError: If dimensions, format, rotation, or buffer layout are invalid.
            ResourceError: If there is an error in creating the image stream.
        """
        self._handle = None
        self._data_owner = None
        self.width = _normalize_int32(width, "width", positive=True)
        self.height = _normalize_int32(height, "height", positive=True)
        self.rotate = _normalize_rotation(rotation)
        self.data_format = _normalize_stream_format(stream_format)
        if self.data_format in _YUV420_STREAM_FORMATS and (self.width % 2 or self.height % 2):
            raise InvalidInputError(
                "YUV420 image width and height must be even",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                width=self.width,
                height=self.height,
                stream_format=self.data_format,
            )
        required_bytes = _required_image_bytes(self.width, self.height, self.data_format)
        if required_bytes > _INT32_MAX:
            raise InvalidInputError(
                "Image buffer size exceeds the native API limit",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                required_bytes=required_bytes,
            )

        if isinstance(data, np.ndarray):
            self._validate_ndarray_layout(data)
        owner, data_ptr, available_bytes = self._prepare_data(data)
        if available_bytes is not None and available_bytes < required_bytes:
            raise InvalidInputError(
                "Image buffer is smaller than required by its dimensions and format",
                errcode.HERR_INVALID_BUFFER_SIZE,
                required_bytes=required_bytes,
                available_bytes=available_bytes,
                width=self.width,
                height=self.height,
                stream_format=self.data_format
            )

        self._data_owner = owner
        image_struct = HFImageData()
        image_struct.data = data_ptr
        image_struct.width = self.width
        image_struct.height = self.height
        image_struct.format = self.data_format
        image_struct.rotation = self.rotate
        handle = HFImageStream()
        ret = HFCreateImageStream(ctypes.byref(image_struct), ctypes.byref(handle))
        check_error(ret, "Create ImageStream", width=self.width, height=self.height, format=self.data_format)
        self._handle = handle

    def _validate_ndarray_layout(self, data):
        if data.dtype != np.uint8:
            raise InvalidInputError(
                "Image ndarray data must be uint8",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                actual_dtype=str(data.dtype)
            )
        if data.ndim == 1:
            return

        if self.data_format in _PACKED_STREAM_CHANNELS:
            channels = _PACKED_STREAM_CHANNELS[self.data_format]
            valid_shapes = {(self.height, self.width, channels)}
            if channels == 1:
                valid_shapes.add((self.height, self.width))
            if data.shape not in valid_shapes:
                raise InvalidInputError(
                    "Image ndarray shape does not match dimensions and format",
                    errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                    actual_shape=data.shape,
                    width=self.width,
                    height=self.height,
                    stream_format=self.data_format
                )
            return

        expected_shape = (self.height * 3 // 2, self.width)
        if data.shape != expected_shape:
            raise InvalidInputError(
                "YUV420 ndarray must be flat or have shape (height * 3 / 2, width)",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                actual_shape=data.shape,
                expected_shape=expected_shape,
                stream_format=self.data_format
            )

    @staticmethod
    def _prepare_data(data):
        if isinstance(data, np.ndarray):
            owner = np.ascontiguousarray(data)
            pointer = owner.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            return owner, pointer, owner.nbytes

        if isinstance(data, bytes):
            pointer = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))
            return data, pointer, len(data)

        pointer_base = getattr(ctypes, "_Pointer", None)
        is_ctypes_pointer = (
            isinstance(data, (ctypes.c_void_p, ctypes.c_char_p))
            or (pointer_base is not None and isinstance(data, pointer_base))
        )
        if isinstance(data, ctypes.Array) or is_ctypes_pointer:
            pointer = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))
            if not pointer:
                raise InvalidInputError(
                    "Image data pointer must not be null",
                    errcode.HERR_INVALID_IMAGE_STREAM_PARAM
                )
            available_bytes = ctypes.sizeof(data) if isinstance(data, ctypes.Array) else None
            return data, pointer, available_bytes

        try:
            view = memoryview(data)
        except TypeError:
            raise InvalidInputError(
                "Image data must support the buffer protocol or be a ctypes pointer",
                errcode.HERR_INVALID_IMAGE_STREAM_PARAM,
                input_type=type(data).__name__
            )

        if not view.c_contiguous:
            owner = view.tobytes()
            pointer = ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8))
            return owner, pointer, len(owner)

        try:
            byte_view = view.cast("B")
        except TypeError:
            owner = view.tobytes()
            pointer = ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8))
            return owner, pointer, len(owner)

        if byte_view.readonly:
            owner = byte_view.tobytes()
            pointer = ctypes.cast(owner, ctypes.POINTER(ctypes.c_uint8))
            return owner, pointer, len(owner)

        owner = np.frombuffer(byte_view, dtype=np.uint8)
        pointer = owner.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        return owner, pointer, owner.nbytes

    def _require_open(self):
        if self._handle is None:
            raise ResourceError(
                "ImageStream has been released",
                errcode.HERR_INVALID_IMAGE_STREAM_HANDLE
            )

    def write_to_file(self, file_path: str):
        """
        Write the image stream to a file. Like PATH/image.jpg
        """
        self._require_open()
        ret = HFDeBugImageStreamDecodeSave(self._handle, file_path)
        check_error(ret, "Write ImageStream to file", file_path=file_path)

    def release(self):
        """
        Release the resources associated with the ImageStream.

        Logs an error if the release fails.
        """
        handle = self._handle
        if handle is not None:
            self._handle = None
            try:
                ret = HFReleaseImageStream(handle)
            finally:
                self._data_owner = None
            check_error(ret, "Release ImageStream")

    def __del__(self):
        """
        Ensure that resources are released when the ImageStream object is garbage collected.
        """
        try:
            self.release()
        except Exception:
            pass

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False

    def debug_show(self):
        """
        Display the image using a debug function provided by the library.
        """
        self._require_open()
        HFDeBugImageStreamImShow(self._handle)

    @property
    def handle(self):
        """
        Return the internal handle of the image stream.
        Returns:
        The handle to the internal image stream, used for interfacing with the underlying C/C++ library.
        """
        return self._handle


# == Session API ==

@dataclass
class FaceExtended:
    """
    A data class to hold extended face information with confidence levels for various attributes.

    Attributes:
        rgb_liveness_confidence (float): Confidence level of RGB-based liveness detection.
        mask_confidence (float): Confidence level of mask detection on the face.
        quality_confidence (float): Confidence level of the overall quality of the face capture.
    """
    rgb_liveness_confidence: float
    mask_confidence: float
    quality_confidence: float
    left_eye_status_confidence: float
    right_eye_status_confidence: float
    action_normal: int
    action_jaw_open: int
    action_shake: int
    action_blink: int
    action_head_raise: int
    race: int
    gender: int
    age_bracket: int
    emotion: int

    def __repr__(self) -> str:
        return f"FaceExtended(rgb_liveness_confidence={self.rgb_liveness_confidence}, mask_confidence={self.mask_confidence}, quality_confidence={self.quality_confidence}, left_eye_status_confidence={self.left_eye_status_confidence}, right_eye_status_confidence={self.right_eye_status_confidence}, action_normal={self.action_normal}, action_jaw_open={self.action_jaw_open}, action_shake={self.action_shake}, action_blink={self.action_blink}, action_head_raise={self.action_head_raise}, race={self.race}, gender={self.gender}, age_bracket={self.age_bracket}, emotion={self.emotion})"


class FaceInformation:
    """
    Holds detailed information about a detected face including location and orientation.

    Attributes:
        track_id (int): Unique identifier for tracking the face across frames.
        location (Tuple): Coordinates of the face in the form (x, y, width, height).
        roll (float): Roll angle of the face.
        yaw (float): Yaw angle of the face.
        pitch (float): Pitch angle of the face.
        _token (HFFaceBasicToken): A token containing low-level details about the face.
        _feature (np.array, optional): An optional numpy array holding the facial feature data.

    Methods:
        __init__: Initializes a new instance of FaceInformation.
    """

    def __init__(self,
                 track_id: int,
                 track_count: int,
                 detection_confidence: float,
                 location: Tuple,
                 roll: float,
                 yaw: float,
                 pitch: float,
                 _token: HFFaceBasicToken,
                 _feature: np.array = None):
        self.track_id = track_id
        self.track_count = track_count
        self.detection_confidence = detection_confidence
        self.location = location
        self.roll = roll
        self.yaw = yaw
        self.pitch = pitch

        # Calculate the required buffer size for the face token and copy it.
        token_size = HInt32()
        ret = HFGetFaceBasicTokenSize(byref(token_size))
        check_error(ret, "Get face basic token size", track_id=track_id)
        if token_size.value <= 0:
            raise ProcessingError(
                "Native library returned an invalid face token size",
                errcode.HERR_INVALID_FACE_TOKEN,
                token_size=token_size.value,
                track_id=track_id,
            )
        buffer_size = token_size.value
        self.buffer = create_string_buffer(buffer_size)
        ret = HFCopyFaceBasicToken(_token, self.buffer, token_size)
        check_error(ret, "Copy face basic token", track_id=track_id)

        # Store the copied token.
        self._token = HFFaceBasicToken()
        self._token.size = buffer_size
        self._token.data = cast(addressof(self.buffer), c_void_p)

    def __repr__(self) -> str:
        return f"FaceInformation(track_id={self.track_id}, track_count={self.track_count}, detection_confidence={self.detection_confidence}, location={self.location}, roll={self.roll}, yaw={self.yaw}, pitch={self.pitch})"


@dataclass
class SessionCustomParameter:
    """
    A data class for configuring the optional parameters in a face recognition session.

    Attributes are set to False by default and can be enabled as needed.

    Methods:
        _c_struct: Converts the Python attributes to a C-compatible structure for session configuration.
    """
    enable_recognition: bool = False
    enable_liveness: bool = False
    enable_ir_liveness: bool = False
    enable_mask_detect: bool = False
    enable_face_attribute: bool = False
    enable_face_quality: bool = False
    enable_interaction_liveness: bool = False
    enable_detect_mode_landmark: bool = False
    enable_face_pose: bool = False
    enable_face_emotion: bool = False

    def _c_struct(self):
        """
        Creates a C structure from the current state of the instance.

        Returns:
            HFSessionCustomParameter: The corresponding C structure with proper type conversions.
        """
        custom_param = HFSessionCustomParameter(
            enable_recognition=int(self.enable_recognition),
            enable_liveness=int(self.enable_liveness),
            enable_ir_liveness=int(self.enable_ir_liveness),
            enable_mask_detect=int(self.enable_mask_detect),
            enable_face_attribute=int(self.enable_face_attribute),
            enable_face_quality=int(self.enable_face_quality),
            enable_interaction_liveness=int(self.enable_interaction_liveness),
            enable_detect_mode_landmark=int(self.enable_detect_mode_landmark),
            enable_face_pose=int(self.enable_face_pose),
            enable_face_emotion=int(self.enable_face_emotion)
        )

        return custom_param

    def __repr__(self) -> str:
        return f"SessionCustomParameter(enable_recognition={self.enable_recognition}, enable_liveness={self.enable_liveness}, enable_ir_liveness={self.enable_ir_liveness}, enable_mask_detect={self.enable_mask_detect}, enable_face_attribute={self.enable_face_attribute}, enable_face_quality={self.enable_face_quality}, enable_interaction_liveness={self.enable_interaction_liveness}, enable_detect_mode_landmark={self.enable_detect_mode_landmark}, enable_face_pose={self.enable_face_pose}, enable_face_emotion={self.enable_face_emotion})"


class InspireFaceSession(object):
    """
    Manages a session for face detection and recognition processes using the InspireFace library.

    Attributes:
        multiple_faces (HFMultipleFaceData): Stores data about multiple detected faces during the session.
        _sess (HFSession): The handle to the underlying library session.
        param (int or SessionCustomParameter): Configuration parameters or flags for the session.

    """

    def __init__(self, param, detect_mode: int = HF_DETECT_MODE_ALWAYS_DETECT,
                 max_detect_num: int = 10, detect_pixel_level=-1, track_by_detect_mode_fps=-1):
        """
        Initializes a new session with the provided configuration parameters.
        
        Args:
            param (int or SessionCustomParameter): Configuration parameters or flags.
            detect_mode (int): Detection mode to be used (e.g., image-based detection).
            max_detect_num (int): Maximum number of faces to detect.
            
        Raises:
            SystemNotReadyError: If InspireFace is not launched.
            ProcessingError: If session creation fails.
        """
        # Initialize _sess to None first to prevent AttributeError in __del__
        self._sess = None
        self.multiple_faces = None
        self.param = param
        self._max_detect_num = None
        
        # If InspireFace is not initialized, run launch() use Pikachu model
        if not query_launch_status():
            ret = launch()
            if not ret:
                raise SystemNotReadyError("Failed to launch InspireFace automatically")

        self._sess = HFSession()
        
        if isinstance(self.param, SessionCustomParameter):
            ret = HFCreateInspireFaceSession(self.param._c_struct(), detect_mode, max_detect_num, detect_pixel_level,
                                             track_by_detect_mode_fps, self._sess)
        elif isinstance(self.param, (int, np.integer)) and not isinstance(self.param, (bool, np.bool_)):
            ret = HFCreateInspireFaceSessionOptional(self.param, detect_mode, max_detect_num, detect_pixel_level,
                                                     track_by_detect_mode_fps, self._sess)
        else:
            raise InvalidInputError("Session parameter must be SessionCustomParameter or int", 
                                   context={'param_type': type(self.param).__name__})
        
        check_error(ret, "Create InspireFace session", 
                   detect_mode=detect_mode, max_detect_num=max_detect_num)
        self._max_detect_num = int(max_detect_num)

    @handle_c_api_errors("Face detection")
    def face_detection(self, image) -> List[FaceInformation]:
        """
        Detects faces in the given image and returns a list of FaceInformation objects containing detailed face data.
        
        Args:
            image (np.ndarray or ImageStream): The image in which to detect faces.
            
        Returns:
            List[FaceInformation]: A list of detected face information.
            
        Raises:
            ResourceError: If session is not initialized.
            ProcessingError: If face detection fails.
        """
        validate_session_initialized(self, "Face detection")
        with self._managed_image_stream(image) as stream:
            self.multiple_faces = HFMultipleFaceData()
            ret = HFExecuteFaceTrack(self._sess, stream.handle,
                                     PHFMultipleFaceData(self.multiple_faces))
            check_error(ret, "Execute face tracking")
            self._validate_detection_results()

            if self.multiple_faces.detectedNum > 0:
                boxes = self._get_faces_boundary_boxes()
                track_ids = self._get_faces_track_ids()
                euler_angle = self._get_faces_euler_angle()
                tokens = self._get_faces_tokens()
                track_counts = self._get_faces_track_counts()

                infos = list()
                for idx in range(self.multiple_faces.detectedNum):
                    top_left = (boxes[idx][0], boxes[idx][1])
                    bottom_right = (boxes[idx][0] + boxes[idx][2], boxes[idx][1] + boxes[idx][3])
                    roll = euler_angle[idx][0]
                    yaw = euler_angle[idx][1]
                    pitch = euler_angle[idx][2]
                    track_id = track_ids[idx]
                    _token = tokens[idx]
                    detection_confidence = self.multiple_faces.detConfidence[idx]
                    track_count = track_counts[idx]

                    info = FaceInformation(
                        location=(top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
                        roll=roll,
                        yaw=yaw,
                        pitch=pitch,
                        track_id=track_id,
                        track_count=track_count,
                        _token=_token,
                        detection_confidence=detection_confidence,
                    )
                    infos.append(info)

                return infos
            return []
        
    def get_face_five_key_points(self, single_face: FaceInformation):
        """Get five key points for a detected face"""
        validate_session_initialized(self, "Get face five key points")
        if not isinstance(single_face, FaceInformation):
            raise InvalidInputError("single_face must be FaceInformation", errcode.HERR_INVALID_FACE_TOKEN)
        num_landmarks = 5
        landmarks_array = (HPoint2f * num_landmarks)()
        ret = HFGetFaceFiveKeyPointsFromFaceToken(single_face._token, landmarks_array, num_landmarks)
        check_error(ret, "Get face five key points", track_id=single_face.track_id)

        landmark = []
        for point in landmarks_array:
            landmark.append(point.x)
            landmark.append(point.y)

        return np.asarray(landmark).reshape(-1, 2)

    def get_face_dense_landmark(self, single_face: FaceInformation):
        """Get dense landmarks for a detected face"""
        validate_session_initialized(self, "Get face dense landmark")
        if not isinstance(single_face, FaceInformation):
            raise InvalidInputError("single_face must be FaceInformation", errcode.HERR_INVALID_FACE_TOKEN)
        num_landmarks = HInt32()
        ret = HFGetNumOfFaceDenseLandmark(byref(num_landmarks))
        check_error(ret, "Get number of dense landmarks")
        if num_landmarks.value <= 0:
            raise ProcessingError(
                "Native library returned an invalid dense landmark count",
                errcode.HERR_SESS_LANDMARK_NOT_ENABLE,
                landmark_count=num_landmarks.value,
            )
        landmarks_array = (HPoint2f * num_landmarks.value)()
        ret = HFGetFaceDenseLandmarkFromFaceToken(single_face._token, landmarks_array, num_landmarks)
        check_error(ret, "Get face dense landmark", track_id=single_face.track_id)

        landmark = []
        for point in landmarks_array:
            landmark.append(point.x)
            landmark.append(point.y)

        return np.asarray(landmark).reshape(-1, 2)
    
    def print_track_cost_spend(self):
        """Print tracking cost statistics"""
        validate_session_initialized(self, "Print track cost spend")
        ret = HFSessionPrintTrackCostSpend(self._sess)
        check_error(ret, "Print track cost spend")

    def set_enable_track_cost_spend(self, enable: bool):
        """Enable or disable track cost spend monitoring"""
        validate_session_initialized(self, "Set enable track cost spend")
        ret = HFSessionSetEnableTrackCostSpend(self._sess, enable)
        check_error(ret, "Set enable track cost spend", enable=enable)
    
    def set_detection_confidence_threshold(self, threshold: float):
        """
        Sets the detection confidence threshold for the face detection session.

        Args:
            threshold (float): The confidence threshold for face detection.
        """
        validate_session_initialized(self, "Set detection confidence threshold")
        ret = HFSessionSetFaceDetectThreshold(self._sess, threshold)
        check_error(ret, "Set detection confidence threshold", threshold=threshold)

    def set_track_preview_size(self, size=192):
        """
        Sets the preview size for the face tracking session.

        Args:
            size (int, optional): The size of the preview area for face tracking. Default is 192.
        """
        validate_session_initialized(self, "Set track preview size")
        ret = HFSessionSetTrackPreviewSize(self._sess, size)
        check_error(ret, "Set track preview size", size=size)

    def set_filter_minimum_face_pixel_size(self, min_size=32):
        """Set minimum face pixel size filter"""
        validate_session_initialized(self, "Set filter minimum face pixel size")
        ret = HFSessionSetFilterMinimumFacePixelSize(self._sess, min_size)
        check_error(ret, "Set filter minimum face pixel size", min_size=min_size)

    def set_track_mode_smooth_ratio(self, ratio=0.025):
        """Set track mode smooth ratio"""
        validate_session_initialized(self, "Set track mode smooth ratio")
        ret = HFSessionSetTrackModeSmoothRatio(self._sess, ratio)
        check_error(ret, "Set track mode smooth ratio", ratio=ratio)

    def set_track_mode_num_smooth_cache_frame(self, num=15):
        """Set track mode number of smooth cache frames"""
        validate_session_initialized(self, "Set track mode num smooth cache frame")
        ret = HFSessionSetTrackModeNumSmoothCacheFrame(self._sess, num)
        check_error(ret, "Set track mode num smooth cache frame", num=num)

    def set_track_model_detect_interval(self, num=20):
        """Set track model detect interval"""
        validate_session_initialized(self, "Set track model detect interval")
        ret = HFSessionSetTrackModeDetectInterval(self._sess, num)
        check_error(ret, "Set track model detect interval", num=num)

    def set_landmark_augmentation_num(self, num=1):
        """Set landmark augmentation number"""
        validate_session_initialized(self, "Set landmark augmentation num")
        ret = HFSessionSetLandmarkAugmentationNum(self._sess, num)
        check_error(ret, "Set landmark augmentation num", num=num)

    def set_track_lost_recovery_mode(self, value=False):
        """Set track lost recovery mode"""
        validate_session_initialized(self, "Set track lost recovery mode")
        ret = HFSessionSetTrackLostRecoveryMode(self._sess, value)
        check_error(ret, "Set track lost recovery mode", value=value)

    @handle_c_api_errors("Face pipeline processing")
    def face_pipeline(self, image, faces: List[FaceInformation], exec_param) -> List[FaceExtended]:
        """
        Processes detected faces to extract additional attributes based on the provided execution parameters.

        Args:
            image (np.ndarray or ImageStream): The image from which faces are detected.
            faces (List[FaceInformation]): A list of FaceInformation objects containing detected face data.
            exec_param (SessionCustomParameter or int): Custom parameters for processing faces.

        Returns:
            List[FaceExtended]: A list of FaceExtended objects with updated attributes like mask confidence, liveness, etc.
        """
        validate_session_initialized(self, "Face pipeline processing")
        if not isinstance(faces, (list, tuple)) or not all(isinstance(face, FaceInformation) for face in faces):
            raise InvalidInputError("faces must be a sequence of FaceInformation", errcode.HERR_INVALID_FACE_LIST)
        if len(faces) == 0:
            return []
        if len(faces) > _INT32_MAX:
            raise InvalidInputError("faces exceeds the native API limit", errcode.HERR_INVALID_FACE_LIST)
        with self._managed_image_stream(image) as stream:
            fn, pm, flag = self._get_processing_function_and_param(exec_param)
            tokens = [face._token for face in faces]
            tokens_array = (HFFaceBasicToken * len(tokens))(*tokens)
            tokens_ptr = cast(tokens_array, PHFFaceBasicToken)

            multi_faces = HFMultipleFaceData()
            multi_faces.detectedNum = len(tokens)
            multi_faces.tokens = tokens_ptr
            ret = fn(self._sess, stream.handle, PHFMultipleFaceData(multi_faces), pm)

            check_error(ret, "Face pipeline processing", num_faces=len(faces))

            extends = [FaceExtended(-1.0, -1.0, -1.0, -1.0, -1.0, 0, 0, 0, 0, 0, -1, -1, -1, -1) for _ in range(len(faces))]
            self._update_mask_confidence(exec_param, flag, extends)
            self._update_rgb_liveness_confidence(exec_param, flag, extends)
            self._update_face_quality_confidence(exec_param, flag, extends)
            self._update_face_attribute_confidence(exec_param, flag, extends)
            self._update_face_interact_confidence(exec_param, flag, extends)
            self._update_face_emotion_confidence(exec_param, flag, extends)

            return extends

    @handle_c_api_errors("Face feature extraction")
    def face_feature_extract(self, image, face_information: FaceInformation):
        """
        Extracts facial features from a specified face within an image for recognition or comparison purposes.

        Args:
            image (np.ndarray or ImageStream): The image from which the face features are to be extracted.
            face_information (FaceInformation): The FaceInformation object containing the details of the face.

        Returns:
            np.ndarray: A numpy array containing the extracted facial features, or None if the extraction fails.
        """
        validate_session_initialized(self, "Face feature extraction")
        if not isinstance(face_information, FaceInformation):
            raise InvalidInputError("face_information must be FaceInformation", errcode.HERR_INVALID_FACE_TOKEN)
        with self._managed_image_stream(image) as stream:
            feature_length = HInt32()
            ret = HFGetFeatureLength(byref(feature_length))
            check_error(ret, "Get face feature length")
            if feature_length.value <= 0:
                raise ProcessingError(
                    "Native library returned an invalid feature length",
                    errcode.HERR_INVALID_FACE_FEATURE,
                    feature_length=feature_length.value,
                )

            feature = np.zeros((feature_length.value,), dtype=np.float32)
            ret = HFFaceFeatureExtractCpy(self._sess, stream.handle, face_information._token,
                                          feature.ctypes.data_as(ctypes.POINTER(HFloat)))

            check_error(ret, "Face feature extraction", track_id=face_information.track_id)
            return feature

    @staticmethod
    def _get_image_stream(image):
        """Convert image to ImageStream if needed"""
        if isinstance(image, np.ndarray):
            return ImageStream.load_from_cv_image(image)
        elif isinstance(image, ImageStream):
            return image
        else:
            raise InvalidInputError("Image must be numpy.ndarray or ImageStream", 
                                   context={'input_type': type(image).__name__})

    @staticmethod
    @contextmanager
    def _managed_image_stream(image):
        """Yield an open stream and deterministically release wrapper-owned streams."""
        if isinstance(image, np.ndarray):
            with ImageStream.load_from_cv_image(image) as stream:
                yield stream
            return
        if isinstance(image, ImageStream):
            image._require_open()
            yield image
            return
        raise InvalidInputError(
            "Image must be numpy.ndarray or ImageStream",
            errcode.HERR_INVALID_PARAM,
            input_type=type(image).__name__,
        )

    @staticmethod
    def _validate_pipeline_result(ret, result, pointer_fields, expected_count, operation):
        check_error(ret, operation)
        actual_count = int(result.num)
        if actual_count != expected_count:
            raise ProcessingError(
                f"{operation} returned an inconsistent result count",
                errcode.HERR_SESS_PIPELINE_FAILURE,
                expected_count=expected_count,
                actual_count=actual_count,
            )
        if actual_count > 0:
            missing = [name for name in pointer_fields if not bool(getattr(result, name))]
            if missing:
                raise ProcessingError(
                    f"{operation} returned null result data",
                    errcode.HERR_SESS_PIPELINE_FAILURE,
                    missing_fields=missing,
                    result_count=actual_count,
                )
        return actual_count

    @staticmethod
    def _get_processing_function_and_param(exec_param):
        """Get processing function and parameters"""
        if isinstance(exec_param, SessionCustomParameter):
            return HFMultipleFacePipelineProcess, exec_param._c_struct(), "object"
        elif isinstance(exec_param, (int, np.integer)) and not isinstance(exec_param, (bool, np.bool_)):
            return HFMultipleFacePipelineProcessOptional, exec_param, "bitmask"
        else:
            raise InvalidInputError("exec_param must be SessionCustomParameter or int",
                                   context={'param_type': type(exec_param).__name__})

    def _update_mask_confidence(self, exec_param, flag, extends):
        """Update mask confidence in extends list"""
        if (flag == "object" and exec_param.enable_mask_detect) or (
                flag == "bitmask" and exec_param & HF_ENABLE_MASK_DETECT):
            mask_results = HFFaceMaskConfidence()
            ret = HFGetFaceMaskConfidence(self._sess, PHFFaceMaskConfidence(mask_results))
            count = self._validate_pipeline_result(ret, mask_results, ("confidence",), len(extends), "Get mask confidence")
            for i in range(count):
                extends[i].mask_confidence = mask_results.confidence[i]

    def _update_face_interact_confidence(self, exec_param, flag, extends):
        """Update face interaction confidence in extends list"""
        if (flag == "object" and exec_param.enable_interaction_liveness) or (
                flag == "bitmask" and exec_param & HF_ENABLE_INTERACTION):
            results = HFFaceInteractionState()
            ret = HFGetFaceInteractionStateResult(self._sess, PHFFaceInteractionState(results))
            count = self._validate_pipeline_result(
                ret,
                results,
                ("leftEyeStatusConfidence", "rightEyeStatusConfidence"),
                len(extends),
                "Get face interaction state",
            )
            for i in range(count):
                extends[i].left_eye_status_confidence = results.leftEyeStatusConfidence[i]
                extends[i].right_eye_status_confidence = results.rightEyeStatusConfidence[i]
                
            actions = HFFaceInteractionsActions()
            ret = HFGetFaceInteractionActionsResult(self._sess, PHFFaceInteractionsActions(actions))
            count = self._validate_pipeline_result(
                ret,
                actions,
                ("normal", "shake", "jawOpen", "headRaise", "blink"),
                len(extends),
                "Get face interaction actions",
            )
            for i in range(count):
                extends[i].action_normal = actions.normal[i]
                extends[i].action_shake = actions.shake[i]
                extends[i].action_jaw_open = actions.jawOpen[i]
                extends[i].action_head_raise = actions.headRaise[i]
                extends[i].action_blink = actions.blink[i]

    def _update_face_emotion_confidence(self, exec_param, flag, extends):
        """Update face emotion confidence in extends list"""
        if (flag == "object" and exec_param.enable_face_emotion) or (
                flag == "bitmask" and exec_param & HF_ENABLE_FACE_EMOTION):
            emotion_results = HFFaceEmotionResult()
            ret = HFGetFaceEmotionResult(self._sess, PHFFaceEmotionResult(emotion_results))
            count = self._validate_pipeline_result(ret, emotion_results, ("emotion",), len(extends), "Get face emotion result")
            for i in range(count):
                extends[i].emotion = emotion_results.emotion[i]

    def _update_rgb_liveness_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        """Update RGB liveness confidence in extends list"""
        if (flag == "object" and exec_param.enable_liveness) or (
                flag == "bitmask" and exec_param & HF_ENABLE_LIVENESS):
            liveness_results = HFRGBLivenessConfidence()
            ret = HFGetRGBLivenessConfidence(self._sess, PHFRGBLivenessConfidence(liveness_results))
            count = self._validate_pipeline_result(ret, liveness_results, ("confidence",), len(extends), "Get RGB liveness confidence")
            for i in range(count):
                extends[i].rgb_liveness_confidence = liveness_results.confidence[i]

    def _update_face_attribute_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        """Update face attribute confidence in extends list"""
        if (flag == "object" and exec_param.enable_face_attribute) or (
                flag == "bitmask" and exec_param & HF_ENABLE_FACE_ATTRIBUTE):
            attribute_results = HFFaceAttributeResult()
            ret = HFGetFaceAttributeResult(self._sess, PHFFaceAttributeResult(attribute_results))
            count = self._validate_pipeline_result(
                ret,
                attribute_results,
                ("gender", "ageBracket", "race"),
                len(extends),
                "Get face attribute result",
            )
            for i in range(count):
                extends[i].gender = attribute_results.gender[i]
                extends[i].age_bracket = attribute_results.ageBracket[i]
                extends[i].race = attribute_results.race[i]

    def _update_face_quality_confidence(self, exec_param, flag, extends: List[FaceExtended]):
        """Update face quality confidence in extends list"""
        if (flag == "object" and exec_param.enable_face_quality) or (
                flag == "bitmask" and exec_param & HF_ENABLE_QUALITY):
            quality_results = HFFaceQualityConfidence()
            ret = HFGetFaceQualityConfidence(self._sess, PHFFaceQualityConfidence(quality_results))
            count = self._validate_pipeline_result(ret, quality_results, ("confidence",), len(extends), "Get face quality confidence")
            for i in range(count):
                extends[i].quality_confidence = quality_results.confidence[i]

    def _validate_detection_results(self):
        count = int(self.multiple_faces.detectedNum)
        if count < 0 or (self._max_detect_num is not None and count > self._max_detect_num):
            raise ProcessingError(
                "Face detection returned an invalid result count",
                errcode.HERR_SESS_TRACKER_FAILURE,
                detected_count=count,
                max_detect_count=self._max_detect_num,
            )
        if count == 0:
            return
        required = {
            "rects": self.multiple_faces.rects,
            "trackIds": self.multiple_faces.trackIds,
            "trackCounts": self.multiple_faces.trackCounts,
            "detConfidence": self.multiple_faces.detConfidence,
            "tokens": self.multiple_faces.tokens,
            "angles.roll": self.multiple_faces.angles.roll,
            "angles.yaw": self.multiple_faces.angles.yaw,
            "angles.pitch": self.multiple_faces.angles.pitch,
        }
        missing = [name for name, pointer in required.items() if not bool(pointer)]
        if missing:
            raise ProcessingError(
                "Face detection returned null result data",
                errcode.HERR_SESS_TRACKER_FAILURE,
                missing_fields=missing,
                detected_count=count,
            )

    def _get_faces_boundary_boxes(self) -> List:
        """Get face boundary boxes from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        rects_ptr = self.multiple_faces.rects
        rects = [(rects_ptr[i].x, rects_ptr[i].y, rects_ptr[i].width, rects_ptr[i].height) for i in range(num_of_faces)]
        return rects

    def _get_faces_track_ids(self) -> List:
        """Get face track IDs from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        track_ids_ptr = self.multiple_faces.trackIds
        track_ids = [track_ids_ptr[i] for i in range(num_of_faces)]
        return track_ids

    def _get_faces_euler_angle(self) -> List:
        """Get face euler angles from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        euler_angle = self.multiple_faces.angles
        angles = [(euler_angle.roll[i], euler_angle.yaw[i], euler_angle.pitch[i]) for i in range(num_of_faces)]
        return angles
     
    def _get_faces_track_counts(self) -> List:
        """Get face track counts from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        track_counts_ptr = self.multiple_faces.trackCounts
        track_counts = [track_counts_ptr[i] for i in range(num_of_faces)]
        return track_counts

    def _get_faces_tokens(self) -> List[HFFaceBasicToken]:
        """Get face tokens from detection results"""
        num_of_faces = self.multiple_faces.detectedNum
        tokens_ptr = self.multiple_faces.tokens
        tokens = [tokens_ptr[i] for i in range(num_of_faces)]
        return tokens

    def release(self):
        """Release session resources"""
        handle = self._sess
        if handle is not None:
            self._sess = None
            ret = HFReleaseInspireFaceSession(handle)
            check_error(ret, "Release InspireFace session")

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __enter__(self):
        validate_session_initialized(self, "Enter session context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


# == Global API ==

def _check_modelscope_availability():
    """
    Check if ModelScope is available when needed and provide helpful error message if not.
    
    Raises ResourceError if ModelScope is needed but not available and OSS is not enabled.
    """
    from .utils.resource import USE_OSS_DOWNLOAD
    
    # Dynamic check for ModelScope availability (don't rely on cached MODELSCOPE_AVAILABLE)
    modelscope_available = True
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as error:
        modelscope_available = False
    
    if not USE_OSS_DOWNLOAD and not modelscope_available:
        raise ResourceError(
            "ModelScope is unavailable; install modelscope or call use_oss_download(True) before downloading models",
            original_error=str(error),
        ) from error

def launch(model_name: str = "Pikachu", resource_path: str = None) -> bool:
    """
    Launches the InspireFace system with the specified resource directory.

    Args:
        model_name (str): the name of the model to use.
        resource_path (str): if None, use the default model path.

    Returns:
        bool: True if the system was successfully launched, False otherwise.

    Raises:
        SystemNotReadyError: If launch fails due to resource issues.
    """
    if resource_path is None:
        from .utils.resource import USE_OSS_DOWNLOAD
        
        # Check if ModelScope is available when needed
        _check_modelscope_availability()
        
        # Use ModelScope by default unless OSS is forced
        sm = ResourceManager(use_modelscope=not USE_OSS_DOWNLOAD)
        resource_path = sm.get_model(model_name, ignore_verification=IGNORE_VERIFICATION_OF_THE_LATEST_MODEL)
    path_c = String(bytes(resource_path, encoding="utf8"))
    ret = HFLaunchInspireFace(path_c)
    if ret != 0:
        if ret == errcode.HERR_ARCHIVE_REPETITION_LOAD:
            logger.warning("Duplicate loading was found")
            return True
        else:
            check_error(ret, "Launch InspireFace", model_name=model_name, resource_path=resource_path)
    return True

def pull_latest_model(model_name: str = "Pikachu") -> str:
    """
    Pulls the latest model from the resource manager.

    Args:
        model_name (str): the name of the model to use.

    Returns:
        str: Path to the downloaded model.
    """
    from .utils.resource import USE_OSS_DOWNLOAD
    
    # Check if ModelScope is available when needed
    _check_modelscope_availability()
    
    sm = ResourceManager(use_modelscope=not USE_OSS_DOWNLOAD)
    resource_path = sm.get_model(model_name, re_download=True)
    return resource_path

def reload(model_name: str = "Pikachu", resource_path: str = None) -> bool:
    """
    Reloads the InspireFace system with the specified resource directory.

    Args:
        model_name (str): the name of the model to use.
        resource_path (str): if None, use the default model path.

    Returns:
        bool: True if reload was successful.
    """
    if resource_path is None:
        from .utils.resource import USE_OSS_DOWNLOAD
        
        # Check if ModelScope is available when needed
        _check_modelscope_availability()
        
        sm = ResourceManager(use_modelscope=not USE_OSS_DOWNLOAD)
        resource_path = sm.get_model(model_name, ignore_verification=IGNORE_VERIFICATION_OF_THE_LATEST_MODEL)
    path_c = String(bytes(resource_path, encoding="utf8"))
    ret = HFReloadInspireFace(path_c)
    if ret != 0:
        if ret == errcode.HERR_ARCHIVE_REPETITION_LOAD:
            logger.warning("Duplicate loading was found")
            return True
        else:
            check_error(ret, "Reload InspireFace", model_name=model_name, resource_path=resource_path)
    return True

def terminate() -> bool:
    """
    Terminates the InspireFace system.

    Returns:
        bool: True if the system was successfully terminated, False otherwise.
    """
    ret = HFTerminateInspireFace()
    check_error(ret, "Terminate InspireFace")
    return True

def query_launch_status() -> bool:
    """
    Queries the launch status of the InspireFace SDK.

    Returns:
        bool: True if InspireFace is launched, False otherwise.
    """
    status = HInt32()
    ret = HFQueryInspireFaceLaunchStatus(byref(status))
    check_error(ret, "Query launch status")
    return status.value == 1

def switch_landmark_engine(engine: int):
    """Switch landmark engine"""
    ret = HFSwitchLandmarkEngine(engine)
    check_error(ret, "Switch landmark engine", engine=engine)
    return True

def switch_image_processing_backend(backend: int):
    """Switch image processing backend"""
    ret = HFSwitchImageProcessingBackend(backend)
    check_error(ret, "Switch image processing backend", backend=backend)
    return True

def set_image_process_aligned_width(width: int):
    """Set the image process aligned width"""
    ret = HFSetImageProcessAlignedWidth(width)
    check_error(ret, "Set image process aligned width", width=width)
    return True

@dataclass
class FeatureHubConfiguration:
    """
    Configuration settings for managing the feature hub, including database and search settings.

    Attributes:
        primary_key_mode (int): Primary key mode for the database.
        enable_persistence (bool): Flag to indicate if the database should be used.
        persistence_db_path (str): Path to the database file.
        search_threshold (float): The threshold value for considering a match.
        search_mode (int): The mode of searching in the database.
    """
    primary_key_mode: int
    enable_persistence: bool
    persistence_db_path: str
    search_threshold: float
    search_mode: int

    def _c_struct(self):
        """
        Converts the data class attributes to a C-compatible structure for use in the InspireFace SDK.

        Returns:
            HFFeatureHubConfiguration: A C-structure for feature hub configuration.
        """
        return HFFeatureHubConfiguration(
            primaryKeyMode=self.primary_key_mode,
            enablePersistence=int(self.enable_persistence),
            persistenceDbPath=String(bytes(self.persistence_db_path, encoding="utf8")),
            searchThreshold=self.search_threshold,
            searchMode=self.search_mode
        )


def feature_hub_enable(config: FeatureHubConfiguration) -> bool:
    """
    Enables the feature hub with the specified configuration.

    Args:
        config (FeatureHubConfiguration): Configuration settings for the feature hub.

    Returns:
        bool: True if successfully enabled, False otherwise.
    """
    ret = HFFeatureHubDataEnable(config._c_struct())
    check_error(ret, "Enable FeatureHub")
    return True


def feature_hub_disable() -> bool:
    """
    Disables the feature hub.

    Returns:
        bool: True if successfully disabled, False otherwise.
    """
    ret = HFFeatureHubDataDisable()
    check_error(ret, "Disable FeatureHub")
    return True


def feature_comparison(feature1: np.ndarray, feature2: np.ndarray) -> float:
    """
    Compares two facial feature arrays to determine their similarity.

    Args:
        feature1 (np.ndarray): The first feature array.
        feature2 (np.ndarray): The second feature array.

    Returns:
        float: A similarity score, where -1.0 indicates an error during comparison.
    """
    validate_feature_data(feature1, "Feature comparison")
    validate_feature_data(feature2, "Feature comparison")
    faces = [feature1, feature2]
    feats = []
    for face in faces:
        feature = HFFaceFeature()
        data_ptr = face.ctypes.data_as(HPFloat)
        feature.size = HInt32(face.size)
        feature.data = data_ptr
        feats.append(feature)

    comparison_result = HFloat()
    ret = HFFaceComparison(feats[0], feats[1], HPFloat(comparison_result))
    check_error(ret, "Face feature comparison")
    return float(comparison_result.value)


class FaceIdentity(object):
    """
    Represents an identity based on facial features, associating the features with a custom ID and a tag.

    Attributes:
        feature (np.ndarray): The facial features as a numpy array.
        id (int): A custom identifier for the face identity.

    Methods:
        __init__: Initializes a new instance of FaceIdentity.
        from_ctypes: Converts a C structure to a FaceIdentity instance.
        _c_struct: Converts the instance back to a compatible C structure.
    """

    def __init__(self, data: np.ndarray, id: int):
        """
        Initializes a new FaceIdentity instance with facial feature data, a custom identifier, and a tag.

        Args:
            data (np.ndarray): The facial feature data.
            id (int): A custom identifier for tracking or referencing the face identity.
        """
        validate_feature_data(data, "FaceIdentity initialization", allow_empty=(id == -1))
        self.feature = data.copy()
        self.id = id

    def __repr__(self) -> str:
        return f"FaceIdentity(id={self.id}, feature={self.feature})"

    @staticmethod
    def from_ctypes(raw_identity: HFFaceFeatureIdentity):
        """
        Converts a ctypes structure representing a face identity into a FaceIdentity object.

        Args:
            raw_identity (HFFaceFeatureIdentity): The ctypes structure containing the face identity data.

        Returns:
            FaceIdentity: An instance of FaceIdentity with data extracted from the ctypes structure.
        """
        if not bool(raw_identity.feature):
            raise ProcessingError("Native identity returned a null feature", errcode.HERR_INVALID_FACE_FEATURE)
        feature_size = raw_identity.feature.contents.size
        feature_data_ptr = raw_identity.feature.contents.data
        if feature_size <= 0 or not bool(feature_data_ptr):
            raise ProcessingError(
                "Native identity returned invalid feature data",
                errcode.HERR_INVALID_FACE_FEATURE,
                feature_size=feature_size,
            )
        feature_data = np.ctypeslib.as_array(cast(feature_data_ptr, HPFloat), (feature_size,))
        id_ = raw_identity.id

        return FaceIdentity(data=feature_data, id=id_)

    def _c_struct(self):
        """
        Converts this FaceIdentity instance into a C-compatible structure for use with InspireFace APIs.

        Returns:
            HFFaceFeatureIdentity: A C structure representing this face identity.
        """
        feature = HFFaceFeature()
        data_ptr = self.feature.ctypes.data_as(HPFloat)
        feature.size = HInt32(self.feature.size)
        feature.data = data_ptr
        return HFFaceFeatureIdentity(
            id=HFaceId(self.id),
            feature=PHFFaceFeature(feature)
        )


def feature_hub_set_search_threshold(threshold: float):
    """
    Sets the search threshold for face matching in the FeatureHub.

    Args:
        threshold (float): The similarity threshold for determining a match.
    """
    ret = HFFeatureHubFaceSearchThresholdSetting(threshold)
    check_error(ret, "Set FeatureHub search threshold", threshold=threshold)


def feature_hub_face_insert(face_identity: FaceIdentity) -> Tuple[bool, int]:
    """
    Inserts a face identity into the FeatureHub database.

    Args:
        face_identity (FaceIdentity): The face identity to insert.

    Returns:
        Tuple[bool, int]: (True, allocated_id) if the face identity was successfully inserted.
    """
    alloc_id = HFaceId()
    ret = HFFeatureHubInsertFeature(face_identity._c_struct(), HPFaceId(alloc_id))
    check_error(ret, "Insert face feature into FeatureHub", identity_id=face_identity.id)
    return True, int(alloc_id.value)


@dataclass
class SearchResult:
    """
    Represents the result of a face search operation with confidence level and the most similar face identity found.

    Attributes:
        confidence (float): The confidence score of the search result, indicating the similarity.
        similar_identity (FaceIdentity): The face identity that most closely matches the search query.
    """
    confidence: float
    similar_identity: FaceIdentity

    def __repr__(self) -> str:
        return f"SearchResult(confidence={self.confidence}, similar_identity={self.similar_identity})"

def feature_hub_face_search(data: np.ndarray) -> SearchResult:
    """
    Searches for the most similar face identity in the feature hub based on provided facial features.

    Args:
        data (np.ndarray): The facial feature data to search for.

    Returns:
        SearchResult: The search result containing the confidence and the most similar identity found.
    """
    validate_feature_data(data, "FeatureHub face search")
    feature = HFFaceFeature(size=HInt32(data.size), data=data.ctypes.data_as(HPFloat))
    confidence = HFloat()
    most_similar = HFFaceFeatureIdentity()
    ret = HFFeatureHubFaceSearch(feature, HPFloat(confidence), PHFFaceFeatureIdentity(most_similar))
    check_error(ret, "Search face in FeatureHub")
    
    if most_similar.id != -1:
        search_identity = FaceIdentity.from_ctypes(most_similar)
        return SearchResult(confidence=confidence.value, similar_identity=search_identity)
    else:
        none = FaceIdentity(np.zeros(0, dtype=np.float32), most_similar.id)
        return SearchResult(confidence=confidence.value, similar_identity=none)


def feature_hub_face_search_top_k(data: np.ndarray, top_k: int) -> List[Tuple]:
    """
    Searches for the top 'k' most similar face identities in the feature hub based on provided facial features.

    Args:
        data (np.ndarray): The facial feature data to search for.
        top_k (int): The number of top results to retrieve.

    Returns:
        List[Tuple]: A list of tuples, each containing the confidence and custom ID of the top results.
    """
    validate_feature_data(data, "FeatureHub face search top k")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or top_k <= 0 or top_k > _INT32_MAX:
        raise InvalidInputError(
            "top_k must be a positive 32-bit integer",
            errcode.HERR_INVALID_PARAM,
            top_k=top_k,
        )
    top_k = int(top_k)
    feature = HFFaceFeature(size=HInt32(data.size), data=data.ctypes.data_as(HPFloat))
    results = HFSearchTopKResults()
    ret = HFFeatureHubFaceSearchTopK(feature, top_k, PHFSearchTopKResults(results))
    check_error(ret, "Search top-k faces in FeatureHub", top_k=top_k)
    if results.size < 0 or results.size > top_k:
        raise ProcessingError(
            "FeatureHub top-k search returned an invalid result count",
            errcode.HERR_FT_HUB_INVALID_FEATURE,
            top_k=top_k,
            result_count=results.size,
        )
    if results.size > 0 and (not bool(results.confidence) or not bool(results.ids)):
        raise ProcessingError(
            "FeatureHub top-k search returned null result data",
            errcode.HERR_FT_HUB_INVALID_FEATURE,
            result_count=results.size,
        )
    outputs = []
    for idx in range(results.size):
        confidence = results.confidence[idx]
        id_ = results.ids[idx]
        outputs.append((confidence, id_))
    return outputs


def feature_hub_face_update(face_identity: FaceIdentity) -> bool:
    """
    Updates an existing face identity in the feature hub.

    Args:
        face_identity (FaceIdentity): The face identity to update.

    Returns:
        bool: True if the update was successful, False otherwise.

    Notes:
        Logs an error if the update operation fails.
    """
    ret = HFFeatureHubFaceUpdate(face_identity._c_struct())
    check_error(ret, "Update face feature in FeatureHub", identity_id=face_identity.id)
    return True


def feature_hub_face_remove(custom_id: int) -> bool:
    """
    Removes a face identity from the feature hub using its custom ID.

    Args:
        custom_id (int): The custom ID of the face identity to remove.

    Returns:
        bool: True if the face was successfully removed, False otherwise.

    Notes:
        Logs an error if the removal operation fails.
    """
    ret = HFFeatureHubFaceRemove(HFaceId(custom_id))
    check_error(ret, "Remove face feature from FeatureHub", custom_id=custom_id)
    return True


def feature_hub_get_face_identity(custom_id: int):
    """
    Retrieves a face identity from the feature hub using its custom ID.

    Args:
        custom_id (int): The custom ID of the face identity to retrieve.

    Returns:
        FaceIdentity: The face identity retrieved, or None if the operation fails.

    Notes:
        Logs an error if retrieving the face identity fails.
    """
    identify = HFFaceFeatureIdentity()
    ret = HFFeatureHubGetFaceIdentity(HFaceId(custom_id), PHFFaceFeatureIdentity(identify))
    check_error(ret, "Get face identity from FeatureHub", custom_id=custom_id)

    return FaceIdentity.from_ctypes(identify)


def feature_hub_get_face_count() -> int:
    """
    Retrieves the total count of face identities stored in the feature hub.

    Returns:
        int: The count of face identities.

    Notes:
        Logs an error if the operation to retrieve the count fails.
    """
    count = HInt32()
    ret = HFFeatureHubGetFaceCount(HPInt32(count))
    check_error(ret, "Get face count")

    return int(count.value)


def feature_hub_get_face_id_list() -> List[int]:
    """
    Retrieves a list of face IDs from the feature hub.

    Returns:
        List[int]: A list of face IDs.
    """
    ids = HFFeatureHubExistingIds()
    ptr = PHFFeatureHubExistingIds(ids)
    ret = HFFeatureHubGetExistingIds(ptr)
    check_error(ret, "Get face id list from FeatureHub")
    if ids.size < 0 or (ids.size > 0 and not bool(ids.ids)):
        raise ProcessingError(
            "FeatureHub returned invalid ID list data",
            errcode.HERR_FT_HUB_DATABASE_FAILURE,
            result_count=ids.size,
        )
    return [int(ids.ids[i]) for i in range(ids.size)]

def view_table_in_terminal():
    """
    Displays the database table of face identities in the terminal.

    Notes:
        Logs an error if the operation to view the table fails.
    """
    ret = HFFeatureHubViewDBTable()
    check_error(ret, "View DB table")

def get_recommended_cosine_threshold() -> float:
    """
    Retrieves the recommended cosine threshold.
    """
    threshold = HFloat()
    ret = HFGetRecommendedCosineThreshold(byref(threshold))
    check_error(ret, "Get recommended cosine threshold")
    return float(threshold.value)

def get_similarity_converter_config() -> dict:
    """
    Retrieves the similarity converter configuration.
    """
    config = HFSimilarityConverterConfig()
    ret = HFGetCosineSimilarityConverter(PHFSimilarityConverterConfig(config))
    check_error(ret, "Get cosine similarity converter config")
    cfg = { 
        "threshold": config.threshold,
        "middleScore": config.middleScore,
        "steepness": config.steepness,
        "outputMin": config.outputMin,
        "outputMax": config.outputMax
    }
    return cfg

def set_similarity_converter_config(cfg: dict):
    """
    Sets the similarity converter configuration.
    """
    config = HFSimilarityConverterConfig()
    config.threshold = cfg["threshold"]
    config.middleScore = cfg["middleScore"]
    config.steepness = cfg["steepness"]
    config.outputMin = cfg["outputMin"]
    config.outputMax = cfg["outputMax"]
    ret = HFUpdateCosineSimilarityConverter(config)
    check_error(ret, "Update cosine similarity converter config")

def cosine_similarity_convert_to_percentage(similarity: float) -> float:
    """
    Converts a cosine similarity score to a percentage similarity score.
    """
    result = HFloat()
    ret = HFCosineSimilarityConvertToPercentage(HFloat(similarity), HPFloat(result))
    check_error(ret, "Convert cosine similarity to percentage", similarity=similarity)
    return float(result.value)

def version() -> str:
    """
    Retrieves the version of the InspireFace library.

    Returns:
        str: The version string of the library.
    """
    ver = HFInspireFaceVersion()
    ret = HFQueryInspireFaceVersion(PHFInspireFaceVersion(ver))
    check_error(ret, "Query InspireFace version")
    return f"{ver.major}.{ver.minor}.{ver.patch}"


_COMPONENT_TYPES = (
    ("mnn", HF_COMPONENT_MNN),
    ("inspirecv", HF_COMPONENT_INSPIRECV),
    ("eigen", HF_COMPONENT_EIGEN),
    ("sqlite", HF_COMPONENT_SQLITE),
    ("sqlite_vec", HF_COMPONENT_SQLITE_VEC),
    ("nlohmann_json", HF_COMPONENT_NLOHMANN_JSON),
    ("opencv", HF_COMPONENT_OPENCV),
    ("tensorrt", HF_COMPONENT_TENSORRT),
    ("cuda", HF_COMPONENT_CUDA),
    ("rknn", HF_COMPONENT_RKNN),
    ("rga", HF_COMPONENT_RGA),
    ("coreml", HF_COMPONENT_COREML),
)
_COMPONENT_STATES = {
    HF_COMPONENT_VERSION_DISABLED: "disabled",
    HF_COMPONENT_VERSION_KNOWN: "known",
    HF_COMPONENT_VERSION_UNKNOWN: "unknown",
}


def _query_native_text(query, operation: str) -> str:
    required_size = HInt32()
    ret = query(None, 0, HPInt32(required_size))
    check_error(ret, f"{operation} size")
    buffer = create_string_buffer(required_size.value)
    copied_size = HInt32()
    ret = query(buffer, len(buffer), HPInt32(copied_size))
    check_error(ret, operation)
    if copied_size.value != required_size.value:
        raise ProcessingError(
            "Native text size changed during the query",
            errcode.HERR_INVALID_BUFFER_SIZE,
            operation=operation,
            required_size=required_size.value,
            copied_size=copied_size.value,
        )
    return buffer.value.decode("utf-8")


def _component_versions_text() -> str:
    return _query_native_text(HFQueryInspireFaceComponentVersions, "Query component-version text")


def component_versions() -> dict:
    """Return SDK and dependency versions without launching InspireFace.

    Each value has ``state`` (``known``, ``unknown``, or ``disabled``), a
    printable ``version`` when known, and numeric fields that are ``None``
    when no version is available.
    """
    sdk_parts = tuple(int(part) for part in version().split("."))
    versions = {
        "inspireface": {
            "state": "known",
            "version": ".".join(str(part) for part in sdk_parts),
            "major": sdk_parts[0],
            "minor": sdk_parts[1],
            "patch": sdk_parts[2],
        }
    }
    for name, component_type in _COMPONENT_TYPES:
        component = HFComponentVersion()
        ret = HFQueryInspireFaceComponentVersion(component_type, PHFComponentVersion(component))
        check_error(ret, "Query component version", component=name)
        state = _COMPONENT_STATES.get(component.state)
        if state is None:
            raise ProcessingError(
                "Native library returned an invalid component-version state",
                errcode.HERR_UNKNOWN,
                component=name,
                state=component.state,
            )
        known = component.state == HF_COMPONENT_VERSION_KNOWN
        versions[name] = {
            "state": state,
            "version": f"{component.major}.{component.minor}.{component.patch}" if known else None,
            "major": component.major if known else None,
            "minor": component.minor if known else None,
            "patch": component.patch if known else None,
        }
    return versions


def diagnostic_info() -> str:
    """Return a copy-and-paste-friendly SDK and dependency diagnostic line."""
    return _query_native_text(HFQueryInspireFaceDiagnosticInformation, "Query InspireFace diagnostic information")


def set_logging_level(level: int) -> None:
    """
    Sets the logging level of the InspireFace library.

    Args:
        level (int): The level to set the logging to.
    """
    ret = HFSetLogLevel(level)
    check_error(ret, "Set logging level", level=level)

def disable_logging() -> None:
    """
    Disables all logging from the InspireFace library.
    """
    ret = HFLogDisable()
    check_error(ret, "Disable logging")

def show_system_resource_statistics():
    """
    Displays the system resource information.
    """
    ret = HFDeBugShowResourceStatistics()
    check_error(ret, "Show system resource statistics")

def switch_apple_coreml_inference_mode(mode: int):
    """
    Switches the Apple CoreML inference mode.
    """
    ret = HFSetAppleCoreMLInferenceMode(mode)
    check_error(ret, "Set Apple CoreML inference mode", mode=mode)
    return True

def set_expansive_hardware_rockchip_dma_heap_path(path: str):
    """
    Sets the path to the expansive hardware Rockchip DMA heap.
    """
    if not isinstance(path, str) or not path or len(path.encode("utf-8")) >= 256:
        raise InvalidInputError(
            "Rockchip DMA heap path must be a non-empty UTF-8 string shorter than 256 bytes",
            errcode.HERR_INVALID_PARAM,
        )
    ret = HFSetExpansiveHardwareRockchipDmaHeapPath(path)
    check_error(ret, "Set expansive hardware Rockchip DMA heap path", path=path)

def query_expansive_hardware_rockchip_dma_heap_path() -> str:
    """
    Queries the path to the expansive hardware Rockchip DMA heap.
    """
    path = create_string_buffer(256)
    path_pointer = cast(path, POINTER(c_char))
    safe_query = globals().get("HFQueryExpansiveHardwareRockchipDmaHeapPathWithSize")
    if safe_query is not None:
        ret = safe_query(path_pointer, len(path))
    else:
        ret = HFQueryExpansiveHardwareRockchipDmaHeapPath(path_pointer)
    check_error(ret, "Query expansive hardware Rockchip DMA heap path")
    return path.value.decode("utf-8")


def set_cuda_device_id(device_id: int):
    """
    Sets the CUDA device ID.
    """
    ret = HFSetCudaDeviceId(device_id)
    check_error(ret, "Set CUDA device ID", device_id=device_id)

def get_cuda_device_id() -> int:
    """
    Gets the CUDA device ID.
    """
    id = HInt32()
    ret = HFGetCudaDeviceId(byref(id))
    check_error(ret, "Get CUDA device ID")
    return int(id.value)

def print_cuda_device_info():
    """
    Prints the CUDA device information.
    """
    ret = HFPrintCudaDeviceInfo()
    check_error(ret, "Print CUDA device information")
    
def get_num_cuda_devices() -> int:
    """
    Gets the number of CUDA devices.
    """
    num = HInt32()
    ret = HFGetNumCudaDevices(byref(num))
    check_error(ret, "Get number of CUDA devices")
    return int(num.value)

def check_cuda_device_support() -> bool:
    """
    Checks if the CUDA device is supported.
    """
    is_support = HInt32()
    ret = HFCheckCudaDeviceSupport(byref(is_support))
    check_error(ret, "Check CUDA device support")
    return bool(is_support.value)
