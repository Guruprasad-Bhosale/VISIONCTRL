"""
Object detection module for VISIONCTRL.

Provides local visual object detection with backends for MediaPipe Tasks
and synthetic mock detection for deterministic testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("VISIONCTRL.ObjectDetector")


class DetectorStatus(str, Enum):
    """Operational status of the object detector."""
    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"
    SIMULATED = "SIMULATED"


@dataclass
class DetectedObject:
    """Detected visual object in camera frame pixel coordinates."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: int
    timestamp: float

    @classmethod
    def create(
        cls,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        timestamp: Optional[float] = None,
    ) -> "DetectedObject":
        """Factory method to construct a DetectedObject with computed center and area."""
        x1, y1, x2, y2 = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        ts = time.perf_counter() if timestamp is None else timestamp

        return cls(
            class_id=int(class_id),
            class_name=str(class_name),
            confidence=float(confidence),
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            center=(cx, cy),
            area=int(area),
            timestamp=ts,
        )


class ObjectDetectorBackend(ABC):
    """Abstract base class for object detector inference backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """Runs object detection on the provided BGR image frame."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """True if the backend is initialized and ready for inference."""
        pass

    @abstractmethod
    def get_status(self) -> Tuple[DetectorStatus, str]:
        """Returns (DetectorStatus, status_detail_message)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Releases backend resources."""
        pass


class MockObjectDetector(ObjectDetectorBackend):
    """Deterministic mock object detector for testing and synthetic scenes."""

    DEFAULT_SYNTHETIC_SCENE = [
        DetectedObject.create(
            class_id=39,
            class_name="bottle",
            confidence=0.92,
            bbox=(500, 250, 650, 450),
        ),
        DetectedObject.create(
            class_id=41,
            class_name="cup",
            confidence=0.88,
            bbox=(700, 300, 800, 400),
        ),
        DetectedObject.create(
            class_id=63,
            class_name="laptop",
            confidence=0.95,
            bbox=(250, 200, 500, 450),
        ),
    ]

    def __init__(
        self,
        scene_objects: Optional[List[DetectedObject]] = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if scene_objects is not None:
            self._objects = list(scene_objects)
        else:
            self._objects = list(self.DEFAULT_SYNTHETIC_SCENE)
        self._available = True
        self._raise_on_detect: bool = False

    def set_scene(self, objects: List[DetectedObject]) -> None:
        """Sets active objects list."""
        self._objects = list(objects)

    def set_raise_on_detect(self, should_raise: bool) -> None:
        """Toggles exception raising during detect for test validation."""
        self._raise_on_detect = should_raise

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        if self._raise_on_detect:
            raise RuntimeError("MockObjectDetector: Simulated inference failure")

        now = time.perf_counter()
        results: List[DetectedObject] = []
        for obj in self._objects:
            if obj.confidence >= self.confidence_threshold:
                results.append(
                    DetectedObject.create(
                        class_id=obj.class_id,
                        class_name=obj.class_name,
                        confidence=obj.confidence,
                        bbox=obj.bbox,
                        timestamp=now,
                    )
                )
        return results

    def is_available(self) -> bool:
        return self._available

    def get_status(self) -> Tuple[DetectorStatus, str]:
        return (DetectorStatus.SIMULATED, "SYNTHETIC_MOCK")

    def close(self) -> None:
        self._available = False


class MediaPipeObjectDetector(ObjectDetectorBackend):
    """Local object detector backend using MediaPipe Tasks."""

    def __init__(
        self,
        model_path: str = "models/efficientdet_lite0.tflite",
        confidence_threshold: float = 0.25,
        max_results: int = 10,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_results = max_results
        self._detector = None
        self._status: DetectorStatus = DetectorStatus.UNINITIALIZED
        self._status_reason: str = ""

        self._initialize()

    def _initialize(self) -> None:
        if not os.path.exists(self.model_path):
            self._status = DetectorStatus.UNAVAILABLE
            self._status_reason = f"MODEL_NOT_FOUND ({self.model_path})"
            logger.info("MediaPipe ObjectDetector model not found at '%s'. Detector unavailable.", self.model_path)
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=self.confidence_threshold,
                max_results=self.max_results,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._detector = vision.ObjectDetector.create_from_options(options)
            self._status = DetectorStatus.READY
            self._status_reason = os.path.basename(self.model_path)
            logger.info("MediaPipe ObjectDetector successfully initialized with model '%s'.", self.model_path)
        except Exception as e:
            self._status = DetectorStatus.ERROR
            self._status_reason = f"INIT_FAILED ({str(e)})"
            logger.error("Failed to initialize MediaPipe ObjectDetector: %s", e)

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        if self._detector is None or self._status != DetectorStatus.READY:
            return []

        import mediapipe as mp

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self._detector.detect(mp_image)
        results: List[DetectedObject] = []
        now = time.perf_counter()

        for detection in detection_result.detections:
            if not detection.categories:
                continue

            category = detection.categories[0]
            confidence = float(category.score)
            if confidence < self.confidence_threshold:
                continue

            class_id = int(category.index) if hasattr(category, "index") else 0
            class_name = str(category.category_name)

            bbox = detection.bounding_box
            x1 = int(max(0, min(w - 1, bbox.origin_x)))
            y1 = int(max(0, min(h - 1, bbox.origin_y)))
            x2 = int(max(0, min(w - 1, bbox.origin_x + bbox.width)))
            y2 = int(max(0, min(h - 1, bbox.origin_y + bbox.height)))

            results.append(
                DetectedObject.create(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    timestamp=now,
                )
            )

        return results

    def is_available(self) -> bool:
        return self._status == DetectorStatus.READY and self._detector is not None

    def get_status(self) -> Tuple[DetectorStatus, str]:
        return (self._status, self._status_reason)

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
                self._detector = None
        self._status = DetectorStatus.UNAVAILABLE
        self._status_reason = "CLOSED"


@dataclass
class ObjectDetectorConfig:
    """Configuration parameters for ObjectDetector orchestrator."""
    confidence_threshold: float = 0.25
    detection_interval_frames: int = 3
    max_detection_age_frames: int = 6
    input_size: Optional[Tuple[int, int]] = None
    model_path: str = "models/efficientdet_lite0.tflite"


class ObjectDetector:
    """
    Object detector orchestrator managing inference interval cadence, caching,
    coordinate scaling, and exception handling.
    """

    def __init__(
        self,
        config: Optional[ObjectDetectorConfig] = None,
        backend: Optional[ObjectDetectorBackend] = None,
    ) -> None:
        self.config = config or ObjectDetectorConfig()
        if backend is not None:
            self._backend = backend
        else:
            self._backend = MediaPipeObjectDetector(
                model_path=self.config.model_path,
                confidence_threshold=self.config.confidence_threshold,
            )

        self._frame_counter: int = 0
        self._cached_detections: List[DetectedObject] = []
        self._cache_age_frames: int = 0
        self._last_latency_ms: float = 0.0
        self._fps: float = 0.0
        self._last_detect_time: float = 0.0
        self._consecutive_errors: int = 0
        self._error_message: Optional[str] = None

    @property
    def backend(self) -> ObjectDetectorBackend:
        return self._backend

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def detector_fps(self) -> float:
        return self._fps

    @property
    def cache_age_frames(self) -> int:
        return self._cache_age_frames

    def get_status(self) -> Tuple[DetectorStatus, str]:
        """Returns current operational status and model/reason string."""
        if self._error_message is not None:
            return (DetectorStatus.ERROR, self._error_message)
        return self._backend.get_status()

    @staticmethod
    def rescale_detections(
        detections: List[DetectedObject],
        orig_width: int,
        orig_height: int,
        detector_width: int,
        detector_height: int,
    ) -> List[DetectedObject]:
        """Transforms bounding boxes from resized detector space back to native frame coordinates."""
        if detector_width <= 0 or detector_height <= 0:
            return detections

        scale_x = orig_width / float(detector_width)
        scale_y = orig_height / float(detector_height)

        rescaled: List[DetectedObject] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            rx1 = int(round(x1 * scale_x))
            ry1 = int(round(y1 * scale_y))
            rx2 = int(round(x2 * scale_x))
            ry2 = int(round(y2 * scale_y))

            rx1 = max(0, min(orig_width - 1, rx1))
            ry1 = max(0, min(orig_height - 1, ry1))
            rx2 = max(0, min(orig_width - 1, rx2))
            ry2 = max(0, min(orig_height - 1, ry2))

            rescaled.append(
                DetectedObject.create(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=(rx1, ry1, rx2, ry2),
                    timestamp=det.timestamp,
                )
            )
        return rescaled

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Runs object detection with interval cadence, caching, and coordinate restoration.

        Args:
            frame: Native camera frame (BGR).

        Returns:
            List of DetectedObject in frame pixel coordinates.
        """
        self._frame_counter += 1

        if not self._backend.is_available():
            self._cached_detections = []
            return []

        interval = max(1, self.config.detection_interval_frames)
        should_run = ((self._frame_counter - 1) % interval == 0) or not self._cached_detections

        if should_run:
            orig_h, orig_w = frame.shape[:2]
            inference_frame = frame

            det_w, det_h = orig_w, orig_h
            if self.config.input_size is not None:
                det_w, det_h = self.config.input_size
                if (det_w != orig_w or det_h != orig_h) and det_w > 0 and det_h > 0:
                    inference_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)

            t0 = time.perf_counter()
            try:
                raw_detections = self._backend.detect(inference_frame)
                t1 = time.perf_counter()
                self._last_latency_ms = (t1 - t0) * 1000.0

                if self._last_detect_time > 0:
                    delta = t1 - self._last_detect_time
                    if delta > 0:
                        instant_fps = 1.0 / delta
                        self._fps = 0.8 * self._fps + 0.2 * instant_fps if self._fps > 0 else instant_fps
                self._last_detect_time = t1

                if (det_w != orig_w or det_h != orig_h):
                    detections = self.rescale_detections(raw_detections, orig_w, orig_h, det_w, det_h)
                else:
                    detections = raw_detections

                self._cached_detections = detections
                self._cache_age_frames = 0
                self._consecutive_errors = 0
                self._error_message = None

            except Exception as e:
                self._consecutive_errors += 1
                self._error_message = f"INFERENCE_ERROR ({str(e)})"
                logger.warning("Object detection exception on frame %d: %s", self._frame_counter, e)
                self._cache_age_frames += 1

                if self._cache_age_frames > self.config.max_detection_age_frames:
                    self._cached_detections = []
        else:
            self._cache_age_frames += 1
            if self._cache_age_frames > self.config.max_detection_age_frames:
                self._cached_detections = []

        return list(self._cached_detections)

    def close(self) -> None:
        """Closes the underlying detector backend."""
        self._cached_detections = []
        self._backend.close()
