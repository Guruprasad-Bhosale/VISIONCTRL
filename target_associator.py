"""
Target association and spatial tracking module for VISIONCTRL.

Associates ProjectedRay 2D camera-plane endpoints with DetectedObjects.
Implements box containment scoring, candidate arbitration, identity continuity,
and temporal activation/release stabilization.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import List, Optional, Tuple

from camera_projection import ProjectedRay
from object_detector import DetectedObject

logger = logging.getLogger("VISIONCTRL.TargetAssociator")


class TargetState(str, Enum):
    """Temporal tracking state of the targeted object."""
    NONE = "NONE"
    SEARCHING = "SEARCHING"
    UNSTABLE = "UNSTABLE"
    LOCKED = "LOCKED"


@dataclass
class TargetedObject:
    """Telemetry representation of the visual object currently targeted by the ray."""
    detected_object: DetectedObject
    score: float
    containment_score: float
    proximity_score: float
    confidence_score: float
    distance_px: float
    inside_bbox: bool
    inside_margin: bool
    state: TargetState
    stable: bool
    frames_active: int
    timestamp: float


@dataclass
class TargetAssociatorConfig:
    """Configuration parameters for target hit-testing and temporal stabilization."""
    target_margin_px: float = 14.0
    activation_frames: int = 3
    release_frames: int = 3
    weight_containment: float = 0.50
    weight_proximity: float = 0.25
    weight_confidence: float = 0.25
    max_proximity_px: float = 300.0
    iou_match_threshold: float = 0.25
    max_center_drift_px: float = 100.0


class TargetAssociator:
    """
    Evaluates 2D spatial intersection between ProjectedRay endpoints and DetectedObjects,
    arbitrates candidate objects, and applies temporal stabilization.
    """

    def __init__(self, config: Optional[TargetAssociatorConfig] = None) -> None:
        self.config = config or TargetAssociatorConfig()
        self._candidate_object: Optional[DetectedObject] = None
        self._candidate_frames: int = 0
        self._active_target: Optional[TargetedObject] = None
        self._release_frames_remaining: int = 0

    def reset(self) -> None:
        """Resets candidate tracking and stabilization history."""
        self._candidate_object = None
        self._candidate_frames = 0
        self._active_target = None
        self._release_frames_remaining = 0

    @property
    def active_target(self) -> Optional[TargetedObject]:
        """Currently active targeted object (if UNSTABLE or LOCKED)."""
        return self._active_target

    @property
    def target_state(self) -> TargetState:
        """Current targeting state."""
        if self._active_target is not None:
            return self._active_target.state
        return TargetState.NONE

    @staticmethod
    def calculate_box_containment(
        px: int,
        py: int,
        bbox: Tuple[int, int, int, int],
        margin: float,
    ) -> Tuple[bool, bool, float]:
        """
        Determines strict bbox containment, expanded margin containment, and
        computes continuous containment score in [0.0, 1.0].

        Returns:
            Tuple of (inside_bbox, inside_margin, containment_score).
        """
        x1, y1, x2, y2 = bbox
        inside_bbox = (x1 <= px <= x2) and (y1 <= py <= y2)

        exp_x1 = x1 - margin
        exp_y1 = y1 - margin
        exp_x2 = x2 + margin
        exp_y2 = y2 + margin
        inside_margin = (exp_x1 <= px <= exp_x2) and (exp_y1 <= py <= exp_y2)

        if not inside_margin:
            return False, False, 0.0

        if inside_bbox:
            dist_to_edge = min(px - x1, x2 - px, py - y1, y2 - py)
            max_inner_dist = max(1.0, min(x2 - x1, y2 - y1) / 2.0)
            inner_closeness = min(1.0, max(0.0, dist_to_edge / max_inner_dist))
            containment_score = 0.5 + 0.5 * inner_closeness
            return True, True, float(containment_score)
        else:
            dx_out = max(0.0, x1 - px, px - x2)
            dy_out = max(0.0, y1 - py, py - y2)
            dist_out = math.sqrt(dx_out * dx_out + dy_out * dy_out)
            margin_closeness = max(0.0, 1.0 - (dist_out / margin)) if margin > 0 else 0.0
            containment_score = 0.5 * margin_closeness
            return False, True, float(containment_score)

    @staticmethod
    def calculate_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        """Calculates 2D Intersection over Union (IoU) between two bounding boxes."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def match_identity(self, obj_a: DetectedObject, obj_b: DetectedObject) -> bool:
        """Lightweight identity continuity check based on class, center distance, and IoU."""
        if obj_a.class_id != obj_b.class_id and obj_a.class_name != obj_b.class_name:
            return False

        cx_a, cy_a = obj_a.center
        cx_b, cy_b = obj_b.center
        center_dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

        if center_dist <= self.config.max_center_drift_px:
            return True

        iou = self.calculate_iou(obj_a.bbox, obj_b.bbox)
        return iou >= self.config.iou_match_threshold

    def evaluate_candidate(
        self,
        ray_endpoint: Tuple[int, int],
        obj: DetectedObject,
    ) -> Optional[Tuple[float, float, float, float, float, bool, bool]]:
        """
        Evaluates an individual DetectedObject against the ray endpoint.

        Returns:
            Tuple of (total_score, containment_score, proximity_score, confidence_score,
                      distance_px, inside_bbox, inside_margin) or None if outside margin.
        """
        px, py = ray_endpoint
        inside_bbox, inside_margin, containment_score = self.calculate_box_containment(
            px, py, obj.bbox, self.config.target_margin_px
        )

        if not inside_margin:
            return None

        cx, cy = obj.center
        distance_px = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        proximity_score = max(0.0, min(1.0, 1.0 - (distance_px / max(1.0, self.config.max_proximity_px))))
        confidence_score = max(0.0, min(1.0, obj.confidence))

        total_score = (
            self.config.weight_containment * containment_score
            + self.config.weight_proximity * proximity_score
            + self.config.weight_confidence * confidence_score
        )
        total_score = max(0.0, min(1.0, total_score))

        return (
            total_score,
            containment_score,
            proximity_score,
            confidence_score,
            distance_px,
            inside_bbox,
            inside_margin,
        )

    def associate_raw(
        self,
        ray_endpoint: Tuple[int, int],
        detected_objects: List[DetectedObject],
    ) -> Optional[Tuple[DetectedObject, float, float, float, float, float, bool, bool]]:
        """
        Performs image-space hit testing and scoring across detected objects.
        Selects candidate with the highest score (tie-broken by proximity, confidence, class_id).
        """
        best_candidate: Optional[Tuple[DetectedObject, float, float, float, float, float, bool, bool]] = None
        best_score = -1.0

        for obj in detected_objects:
            eval_res = self.evaluate_candidate(ray_endpoint, obj)
            if eval_res is None:
                continue

            (
                total_score,
                containment_score,
                proximity_score,
                confidence_score,
                distance_px,
                inside_bbox,
                inside_margin,
            ) = eval_res

            is_better = False
            if total_score > best_score + 1e-5:
                is_better = True
            elif abs(total_score - best_score) <= 1e-5 and best_candidate is not None:
                if distance_px < best_candidate[5] - 1e-3:
                    is_better = True
                elif abs(distance_px - best_candidate[5]) <= 1e-3:
                    if confidence_score > best_candidate[4] + 1e-3:
                        is_better = True
                    elif abs(confidence_score - best_candidate[4]) <= 1e-3:
                        if obj.class_id < best_candidate[0].class_id:
                            is_better = True

            if is_better:
                best_score = total_score
                best_candidate = (
                    obj,
                    total_score,
                    containment_score,
                    proximity_score,
                    confidence_score,
                    distance_px,
                    inside_bbox,
                    inside_margin,
                )

        return best_candidate

    def update(
        self,
        projected_ray: Optional[ProjectedRay],
        detected_objects: List[DetectedObject],
        active_gesture: str = "POINTING",
        pointing_valid: bool = True,
        timestamp: Optional[float] = None,
    ) -> Optional[TargetedObject]:
        """
        Updates targeting association for the current frame.

        Args:
            projected_ray: 2D ProjectedRay from PointingEstimator / CameraProjection.
            detected_objects: List of current DetectedObject from ObjectDetector.
            active_gesture: String representation of active gesture.
            pointing_valid: Boolean indicating whether pointing geometric confidence is valid.
            timestamp: Optional monotonic timestamp.

        Returns:
            TargetedObject if an object is targeted, or None.
        """
        now = time.perf_counter() if timestamp is None else timestamp

        if (
            projected_ray is None
            or not projected_ray.visible
            or not pointing_valid
            or active_gesture != "POINTING"
        ):
            self.reset()
            return None

        ray_endpoint = projected_ray.endpoint_px
        raw_res = self.associate_raw(ray_endpoint, detected_objects)

        if raw_res is not None:
            (
                obj,
                total_score,
                containment_score,
                proximity_score,
                confidence_score,
                distance_px,
                inside_bbox,
                inside_margin,
            ) = raw_res

            if self._candidate_object is not None and self.match_identity(self._candidate_object, obj):
                self._candidate_frames += 1
            else:
                self._candidate_object = obj
                self._candidate_frames = 1

            self._candidate_object = obj

            if self._candidate_frames >= self.config.activation_frames:
                target_state = TargetState.LOCKED
                self._release_frames_remaining = self.config.release_frames
            else:
                target_state = TargetState.UNSTABLE

            self._active_target = TargetedObject(
                detected_object=obj,
                score=total_score,
                containment_score=containment_score,
                proximity_score=proximity_score,
                confidence_score=confidence_score,
                distance_px=distance_px,
                inside_bbox=inside_bbox,
                inside_margin=inside_margin,
                state=target_state,
                stable=(target_state == TargetState.LOCKED),
                frames_active=self._candidate_frames,
                timestamp=now,
            )
            return self._active_target

        else:
            if self._active_target is not None and self._release_frames_remaining > 0:
                self._release_frames_remaining -= 1
                decayed_state = TargetState.LOCKED if self._release_frames_remaining >= 2 else TargetState.UNSTABLE
                self._active_target = TargetedObject(
                    detected_object=self._active_target.detected_object,
                    score=self._active_target.score * 0.90,
                    containment_score=self._active_target.containment_score,
                    proximity_score=self._active_target.proximity_score,
                    confidence_score=self._active_target.confidence_score,
                    distance_px=self._active_target.distance_px,
                    inside_bbox=self._active_target.inside_bbox,
                    inside_margin=self._active_target.inside_margin,
                    state=decayed_state,
                    stable=(decayed_state == TargetState.LOCKED),
                    frames_active=self._candidate_frames,
                    timestamp=now,
                )
                return self._active_target
            else:
                self.reset()
                return None
