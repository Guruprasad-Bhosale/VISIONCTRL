"""
Spatial ray and pointing estimation module for VISIONCTRL.

Transforms index-finger pointing geometry into a continuous 3D spatial ray
in normalized landmark space with temporal vector smoothing and confidence metrics.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import math
from typing import List, Optional, Tuple

from hand_tracker import HandData, HandLandmarkIndex, LandmarkPoint

logger = logging.getLogger("VISIONCTRL.PointingEstimator")


@dataclass
class SpatialRay:
    """3D geometric ray in normalized landmark coordinates."""
    origin_3d: Tuple[float, float, float]
    direction_3d: Tuple[float, float, float]
    length: float
    endpoint_3d: Tuple[float, float, float]
    timestamp: float
    valid: bool


@dataclass
class PointingEstimate:
    """Pointing estimation container with 3D ray and geometric reliability metrics."""
    ray: SpatialRay
    confidence: float
    index_axis_quality: float
    hand_valid: bool
    pointing_valid: bool


class OriginStrategy(str, Enum):
    """Landmark references for ray origin."""
    INDEX_TIP = "INDEX_TIP"
    INDEX_DIP = "INDEX_DIP"
    INDEX_PIP = "INDEX_PIP"
    INDEX_MCP = "INDEX_MCP"


@dataclass
class PointingConfig:
    """Configuration parameters for PointingEstimator."""
    origin_strategy: str = "INDEX_TIP"
    ray_length_units: float = 0.50
    direction_smoothing: float = 0.55
    ray_activate_confidence: float = 0.40
    ray_release_confidence: float = 0.30
    loss_grace_period_frames: int = 3


class PointingEstimator:
    """
    Estimates a continuous 3D spatial pointing ray from index finger geometry.
    Combines finger joint segments, applies directional EMA filtering,
    evaluates geometric confidence, and manages activation hysteresis.
    """

    def __init__(self, config: Optional[PointingConfig] = None) -> None:
        self.config = config or PointingConfig()
        self._prev_direction: Optional[Tuple[float, float, float]] = None
        self._is_active: bool = False
        self._grace_frames_remaining: int = 0
        self._last_valid_ray: Optional[SpatialRay] = None

    def reset(self) -> None:
        """Resets temporal smoothing filters and activation state."""
        self._prev_direction = None
        self._is_active = False
        self._grace_frames_remaining = 0
        self._last_valid_ray = None

    @property
    def is_active(self) -> bool:
        return self._is_active

    @staticmethod
    def _compute_segment_vector(
        pt_start: LandmarkPoint,
        pt_end: LandmarkPoint,
    ) -> Tuple[Tuple[float, float, float], float]:
        """Calculates 3D difference vector and its Euclidean magnitude."""
        dx = pt_end.x - pt_start.x
        dy = pt_end.y - pt_start.y
        dz = pt_end.z - pt_start.z
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        return (dx, dy, dz), mag

    @classmethod
    def compute_finger_axis(
        cls,
        landmarks: List[LandmarkPoint],
    ) -> Tuple[Optional[Tuple[float, float, float]], float]:
        """
        Calculates the 3D finger axis by combining segment vectors with distal weighting.

        Returns:
            Tuple of (unit_direction_vector or None, axis_alignment_quality [0.0, 1.0]).
        """
        mcp = landmarks[HandLandmarkIndex.INDEX_FINGER_MCP]
        pip = landmarks[HandLandmarkIndex.INDEX_FINGER_PIP]
        dip = landmarks[HandLandmarkIndex.INDEX_FINGER_DIP]
        tip = landmarks[HandLandmarkIndex.INDEX_FINGER_TIP]

        v1_raw, mag1 = cls._compute_segment_vector(mcp, pip)
        v2_raw, mag2 = cls._compute_segment_vector(pip, dip)
        v3_raw, mag3 = cls._compute_segment_vector(dip, tip)

        if mag1 < 1e-6 or mag2 < 1e-6 or mag3 < 1e-6:
            return None, 0.0

        v1 = (v1_raw[0] / mag1, v1_raw[1] / mag1, v1_raw[2] / mag1)
        v2 = (v2_raw[0] / mag2, v2_raw[1] / mag2, v2_raw[2] / mag2)
        v3 = (v3_raw[0] / mag3, v3_raw[1] / mag3, v3_raw[2] / mag3)

        cos1 = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        cos2 = v2[0] * v3[0] + v2[1] * v3[1] + v2[2] * v3[2]

        q1 = max(0.0, min(1.0, (cos1 + 1.0) / 2.0))
        q2 = max(0.0, min(1.0, (cos2 + 1.0) / 2.0))
        axis_quality = (q1 + q2) / 2.0

        w1, w2, w3 = 0.20, 0.30, 0.50
        dir_x = w1 * v1[0] + w2 * v2[0] + w3 * v3[0]
        dir_y = w1 * v1[1] + w2 * v2[1] + w3 * v3[1]
        dir_z = w1 * v1[2] + w2 * v2[2] + w3 * v3[2]

        total_mag = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
        if total_mag < 1e-6:
            return None, 0.0

        unit_dir = (dir_x / total_mag, dir_y / total_mag, dir_z / total_mag)
        return unit_dir, axis_quality

    def _get_ray_origin(self, landmarks: List[LandmarkPoint]) -> Tuple[float, float, float]:
        """Extracts 3D ray origin based on configured origin strategy."""
        strategy = self.config.origin_strategy.upper()
        if strategy == OriginStrategy.INDEX_MCP.value:
            pt = landmarks[HandLandmarkIndex.INDEX_FINGER_MCP]
        elif strategy == OriginStrategy.INDEX_PIP.value:
            pt = landmarks[HandLandmarkIndex.INDEX_FINGER_PIP]
        elif strategy == OriginStrategy.INDEX_DIP.value:
            pt = landmarks[HandLandmarkIndex.INDEX_FINGER_DIP]
        else:
            pt = landmarks[HandLandmarkIndex.INDEX_FINGER_TIP]
        return (pt.x, pt.y, pt.z)

    def _calculate_confidence(
        self,
        hand: HandData,
        axis_quality: float,
        raw_direction: Tuple[float, float, float],
    ) -> float:
        """Calculates a deterministic geometric reliability score bounded in [0.0, 1.0]."""
        wrist = hand.wrist
        tip = hand.index_tip
        pip = hand.landmarks[HandLandmarkIndex.INDEX_FINGER_PIP]

        dist_tip_wrist = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2 + (tip.z - wrist.z) ** 2)
        dist_pip_wrist = math.sqrt((pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2 + (pip.z - wrist.z) ** 2)

        ratio = dist_tip_wrist / (dist_pip_wrist + 1e-6)
        finger_ext_score = max(0.0, min(1.0, (ratio - 0.90) / 0.35))
        landmark_score = max(0.0, min(1.0, float(hand.score)))

        if self._prev_direction is not None:
            dot = (
                raw_direction[0] * self._prev_direction[0]
                + raw_direction[1] * self._prev_direction[1]
                + raw_direction[2] * self._prev_direction[2]
            )
            temporal_score = max(0.0, min(1.0, (dot + 1.0) / 2.0))
        else:
            temporal_score = 1.0

        confidence = (
            0.40 * axis_quality
            + 0.30 * finger_ext_score
            + 0.20 * landmark_score
            + 0.10 * temporal_score
        )
        return max(0.0, min(1.0, confidence))

    def estimate(
        self,
        hand: Optional[HandData],
        active_gesture: str,
        timestamp: float,
    ) -> PointingEstimate:
        """
        Calculates pointing estimate from hand landmarks and active gesture state.

        Args:
            hand: Structured HandData from active hand, or None.
            active_gesture: Currently active gesture string.
            timestamp: Monotonic frame timestamp in seconds.

        Returns:
            PointingEstimate container.
        """
        is_pointing_gesture = (str(active_gesture) == "POINTING")

        if hand is None or not hand.landmarks or len(hand.landmarks) != 21:
            if self._is_active and self._grace_frames_remaining > 0 and self._last_valid_ray is not None:
                self._grace_frames_remaining -= 1
                return PointingEstimate(
                    ray=self._last_valid_ray,
                    confidence=0.0,
                    index_axis_quality=0.0,
                    hand_valid=False,
                    pointing_valid=False,
                )
            self.reset()
            empty_ray = SpatialRay(
                origin_3d=(0.0, 0.0, 0.0),
                direction_3d=(0.0, 0.0, 1.0),
                length=self.config.ray_length_units,
                endpoint_3d=(0.0, 0.0, self.config.ray_length_units),
                timestamp=timestamp,
                valid=False,
            )
            return PointingEstimate(
                ray=empty_ray,
                confidence=0.0,
                index_axis_quality=0.0,
                hand_valid=False,
                pointing_valid=False,
            )

        raw_direction, axis_quality = self.compute_finger_axis(hand.landmarks)
        if raw_direction is None:
            self.reset()
            empty_ray = SpatialRay(
                origin_3d=self._get_ray_origin(hand.landmarks),
                direction_3d=(0.0, 0.0, 1.0),
                length=self.config.ray_length_units,
                endpoint_3d=(0.0, 0.0, self.config.ray_length_units),
                timestamp=timestamp,
                valid=False,
            )
            return PointingEstimate(
                ray=empty_ray,
                confidence=0.0,
                index_axis_quality=axis_quality,
                hand_valid=True,
                pointing_valid=False,
            )

        confidence = self._calculate_confidence(hand, axis_quality, raw_direction)

        if not self._is_active:
            if is_pointing_gesture and confidence >= self.config.ray_activate_confidence:
                self._is_active = True
                self._grace_frames_remaining = self.config.loss_grace_period_frames
        else:
            if not is_pointing_gesture or confidence < self.config.ray_release_confidence:
                self._is_active = False
                self._prev_direction = None

        if self._prev_direction is not None and self.config.direction_smoothing > 0.0:
            alpha = max(0.0, min(1.0, self.config.direction_smoothing))
            smooth_x = alpha * raw_direction[0] + (1.0 - alpha) * self._prev_direction[0]
            smooth_y = alpha * raw_direction[1] + (1.0 - alpha) * self._prev_direction[1]
            smooth_z = alpha * raw_direction[2] + (1.0 - alpha) * self._prev_direction[2]
            smooth_mag = math.sqrt(smooth_x * smooth_x + smooth_y * smooth_y + smooth_z * smooth_z)
            if smooth_mag > 1e-6:
                final_dir = (smooth_x / smooth_mag, smooth_y / smooth_mag, smooth_z / smooth_mag)
            else:
                final_dir = raw_direction
        else:
            final_dir = raw_direction

        if self._is_active:
            self._prev_direction = final_dir

        origin_3d = self._get_ray_origin(hand.landmarks)
        length = self.config.ray_length_units
        endpoint_3d = (
            origin_3d[0] + final_dir[0] * length,
            origin_3d[1] + final_dir[1] * length,
            origin_3d[2] + final_dir[2] * length,
        )

        ray = SpatialRay(
            origin_3d=origin_3d,
            direction_3d=final_dir,
            length=length,
            endpoint_3d=endpoint_3d,
            timestamp=timestamp,
            valid=self._is_active,
        )

        if self._is_active:
            self._last_valid_ray = ray
            self._grace_frames_remaining = self.config.loss_grace_period_frames

        return PointingEstimate(
            ray=ray,
            confidence=confidence,
            index_axis_quality=axis_quality,
            hand_valid=True,
            pointing_valid=self._is_active,
        )
