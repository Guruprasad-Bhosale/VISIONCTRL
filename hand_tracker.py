"""
Hand tracking module for VISIONCTRL.

Encapsulates MediaPipe HandLandmarker for real-time hand and landmark tracking.
Provides structured access to 21 3D hand landmarks in normalized and pixel coordinates.
"""

from dataclasses import dataclass
from enum import IntEnum
import logging
import os
import time
import urllib.request
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

logger = logging.getLogger("VISIONCTRL.HandTracker")


class HandLandmarkIndex(IntEnum):
    """Standard 21 MediaPipe Hand Landmark indices."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Palm
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.THUMB_CMC),
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.INDEX_FINGER_MCP),
    (HandLandmarkIndex.INDEX_FINGER_MCP, HandLandmarkIndex.MIDDLE_FINGER_MCP),
    (HandLandmarkIndex.MIDDLE_FINGER_MCP, HandLandmarkIndex.RING_FINGER_MCP),
    (HandLandmarkIndex.RING_FINGER_MCP, HandLandmarkIndex.PINKY_MCP),
    (HandLandmarkIndex.WRIST, HandLandmarkIndex.PINKY_MCP),
    # Thumb
    (HandLandmarkIndex.THUMB_CMC, HandLandmarkIndex.THUMB_MCP),
    (HandLandmarkIndex.THUMB_MCP, HandLandmarkIndex.THUMB_IP),
    (HandLandmarkIndex.THUMB_IP, HandLandmarkIndex.THUMB_TIP),
    # Index finger
    (HandLandmarkIndex.INDEX_FINGER_MCP, HandLandmarkIndex.INDEX_FINGER_PIP),
    (HandLandmarkIndex.INDEX_FINGER_PIP, HandLandmarkIndex.INDEX_FINGER_DIP),
    (HandLandmarkIndex.INDEX_FINGER_DIP, HandLandmarkIndex.INDEX_FINGER_TIP),
    # Middle finger
    (HandLandmarkIndex.MIDDLE_FINGER_MCP, HandLandmarkIndex.MIDDLE_FINGER_PIP),
    (HandLandmarkIndex.MIDDLE_FINGER_PIP, HandLandmarkIndex.MIDDLE_FINGER_DIP),
    (HandLandmarkIndex.MIDDLE_FINGER_DIP, HandLandmarkIndex.MIDDLE_FINGER_TIP),
    # Ring finger
    (HandLandmarkIndex.RING_FINGER_MCP, HandLandmarkIndex.RING_FINGER_PIP),
    (HandLandmarkIndex.RING_FINGER_PIP, HandLandmarkIndex.RING_FINGER_DIP),
    (HandLandmarkIndex.RING_FINGER_DIP, HandLandmarkIndex.RING_FINGER_TIP),
    # Pinky finger
    (HandLandmarkIndex.PINKY_MCP, HandLandmarkIndex.PINKY_PIP),
    (HandLandmarkIndex.PINKY_PIP, HandLandmarkIndex.PINKY_DIP),
    (HandLandmarkIndex.PINKY_DIP, HandLandmarkIndex.PINKY_TIP),
)


@dataclass
class LandmarkPoint:
    """Represents a single 3D hand landmark in normalized and pixel space."""
    x: float
    y: float
    z: float
    px: int
    py: int


@dataclass
class HandData:
    """Structured data container for a detected hand in a frame."""
    handedness: str
    score: float
    landmarks: List[LandmarkPoint]

    @property
    def wrist(self) -> LandmarkPoint:
        return self.landmarks[HandLandmarkIndex.WRIST]

    @property
    def thumb_tip(self) -> LandmarkPoint:
        return self.landmarks[HandLandmarkIndex.THUMB_TIP]

    @property
    def index_tip(self) -> LandmarkPoint:
        return self.landmarks[HandLandmarkIndex.INDEX_FINGER_TIP]

    @property
    def middle_tip(self) -> LandmarkPoint:
        return self.landmarks[HandLandmarkIndex.MIDDLE_FINGER_TIP]

    @property
    def ring_tip(self) -> LandmarkPoint:
        return self.landmarks[HandLandmarkIndex.RING_FINGER_TIP]

    @property
    def pinky_tip(self) -> LandmarkPoint:
        return self.landmarks[HandLandmarkIndex.PINKY_TIP]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Calculates pixel bounding box (min_x, min_y, max_x, max_y)."""
        xs = [lm.px for lm in self.landmarks]
        ys = [lm.py for lm in self.landmarks]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> Tuple[int, int]:
        """Returns approximate center pixel of the hand bounding box."""
        min_x, min_y, max_x, max_y = self.bbox
        return ((min_x + max_x) // 2, (min_y + max_y) // 2)


class HandTracker:
    """
    Manager for MediaPipe HandLandmarker.
    Processes BGR frames and returns structured HandData collections.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._last_timestamp_ms: int = 0
        self._landmarker: Optional[vision.HandLandmarker] = None

        self._ensure_model_exists()
        self._initialize_landmarker()

    def _ensure_model_exists(self) -> None:
        """Verifies local model presence or downloads canonical model asset."""
        if os.path.exists(self.model_path):
            return

        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        logger.info("Model not found at %s. Downloading canonical model asset...", self.model_path)
        try:
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            logger.info("Model downloaded successfully (%d bytes).", os.path.getsize(self.model_path))
        except Exception as e:
            logger.error("Failed to download model asset from %s: %s", self.MODEL_URL, e)
            raise FileNotFoundError(
                f"Model file missing at '{self.model_path}' and automatic download failed: {e}"
            ) from e

    def _initialize_landmarker(self) -> None:
        """Initializes MediaPipe HandLandmarker in VIDEO running mode."""
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker initialized successfully in VIDEO mode.")

    def _get_next_timestamp_ms(self) -> int:
        """Generates strictly monotonic timestamps in milliseconds."""
        current_ms = int(time.time() * 1000)
        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms
        return current_ms

    def process_frame(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Processes a single BGR frame and returns detected hands with landmarks.

        Args:
            frame_bgr: OpenCV image frame in BGR format.

        Returns:
            List of HandData instances for each detected hand.
        """
        if self._landmarker is None or frame_bgr is None or frame_bgr.size == 0:
            return []

        height, width = frame_bgr.shape[:2]
        if height == 0 or width == 0:
            return []

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = self._get_next_timestamp_ms()

            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            if not result or not result.hand_landmarks:
                return []

            detected_hands: List[HandData] = []
            for i, raw_landmarks in enumerate(result.hand_landmarks):
                handedness_label = "Unknown"
                score = 1.0
                if result.handedness and i < len(result.handedness) and result.handedness[i]:
                    category = result.handedness[i][0]
                    handedness_label = category.category_name or category.display_name or "Unknown"
                    score = float(category.score)

                landmarks: List[LandmarkPoint] = []
                for lm in raw_landmarks:
                    px = int(np.clip(lm.x * width, 0, width - 1))
                    py = int(np.clip(lm.y * height, 0, height - 1))
                    landmarks.append(
                        LandmarkPoint(
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(lm.z),
                            px=px,
                            py=py,
                        )
                    )

                detected_hands.append(
                    HandData(
                        handedness=handedness_label,
                        score=score,
                        landmarks=landmarks,
                    )
                )

            return detected_hands

        except Exception as exc:
            logger.warning("Error processing frame in HandTracker: %s", exc)
            return []

    def close(self) -> None:
        """Closes the landmarker and frees resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
