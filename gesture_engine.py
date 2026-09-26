"""
Gesture engine and interaction arbitration module for VISIONCTRL.

Extracts geometric finger features, arbitrates candidate gestures,
applies temporal stabilization, and manages mouse action dispatching.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import Optional, Tuple

from cursor_controller import MockMouseBackend, MouseBackend
from hand_tracker import HandData, HandLandmarkIndex, LandmarkPoint

logger = logging.getLogger("VISIONCTRL.GestureEngine")


class GestureType(str, Enum):
    """Classified hand gestures."""
    POINTING = "POINTING"
    PINCHING = "PINCHING"
    TWO_FINGER = "TWO_FINGER"
    THREE_FINGER = "THREE_FINGER"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    UNKNOWN = "UNKNOWN"


class GestureState(str, Enum):
    """Pinch and drag lifecycle states."""
    POINTING = "POINTING"
    PINCHING = "PINCHING"
    DRAGGING = "DRAGGING"
    RELEASED = "RELEASED"


class MouseActionState(str, Enum):
    """Discrete OS mouse action dispatch states."""
    IDLE = "IDLE"
    CLICKED = "CLICKED"
    RIGHT_CLICKED = "RIGHT_CLICKED"
    MIDDLE_CLICKED = "MIDDLE_CLICKED"
    BUTTON_DOWN = "BUTTON_DOWN"
    BUTTON_UP = "BUTTON_UP"
    LOCKED = "LOCKED"
    PAUSED = "PAUSED"


@dataclass
class FingerExtensionFeatures:
    """Geometric extension states of hand digits."""
    index_extended: bool
    middle_extended: bool
    ring_extended: bool
    pinky_extended: bool
    thumb_extended: bool
    thumb_index_pinch: bool
    confidence: float = 1.0


class HandFeatureExtractor:
    """Extracts geometric features from 3D hand landmarks."""

    @staticmethod
    def _is_digit_extended(
        mcp: LandmarkPoint,
        pip: LandmarkPoint,
        dip: LandmarkPoint,
        tip: LandmarkPoint,
        wrist: LandmarkPoint,
    ) -> bool:
        """Determines if a finger is extended using 3D vector alignment and radial distances."""
        v1 = (pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z)
        v2 = (dip.x - pip.x, dip.y - pip.y, dip.z - pip.z)
        v3 = (tip.x - dip.x, tip.y - dip.y, tip.z - dip.z)

        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) + 1e-6
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2) + 1e-6
        mag3 = math.sqrt(v3[0] ** 2 + v3[1] ** 2 + v3[2] ** 2) + 1e-6

        cos_theta1 = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (mag1 * mag2)
        cos_theta2 = (v2[0] * v3[0] + v2[1] * v3[1] + v2[2] * v3[2]) / (mag2 * mag3)

        dist_tip_wrist = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2 + (tip.z - wrist.z) ** 2)
        dist_pip_wrist = math.sqrt((pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2 + (pip.z - wrist.z) ** 2)
        dist_tip_mcp = math.sqrt((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2 + (tip.z - mcp.z) ** 2)
        dist_pip_mcp = math.sqrt((pip.x - mcp.x) ** 2 + (pip.y - mcp.y) ** 2 + (pip.z - mcp.z) ** 2)

        if dist_tip_wrist < dist_pip_wrist and dist_tip_mcp < dist_pip_mcp:
            return False

        is_straight = (cos_theta1 > 0.55 and cos_theta2 > 0.55)
        is_outward = (dist_tip_wrist > 1.10 * dist_pip_wrist and dist_tip_mcp > 1.20 * dist_pip_mcp)

        return (is_straight and is_outward) or (dist_tip_wrist > 1.18 * dist_pip_wrist)

    @classmethod
    def extract(
        cls,
        hand: HandData,
        normalized_pinch_dist: float,
        pinch_threshold: float = 0.35,
    ) -> FingerExtensionFeatures:
        """Extracts complete feature set for the given hand."""
        lms = hand.landmarks
        wrist = lms[HandLandmarkIndex.WRIST]

        index_ext = cls._is_digit_extended(
            lms[HandLandmarkIndex.INDEX_FINGER_MCP],
            lms[HandLandmarkIndex.INDEX_FINGER_PIP],
            lms[HandLandmarkIndex.INDEX_FINGER_DIP],
            lms[HandLandmarkIndex.INDEX_FINGER_TIP],
            wrist,
        )

        middle_ext = cls._is_digit_extended(
            lms[HandLandmarkIndex.MIDDLE_FINGER_MCP],
            lms[HandLandmarkIndex.MIDDLE_FINGER_PIP],
            lms[HandLandmarkIndex.MIDDLE_FINGER_DIP],
            lms[HandLandmarkIndex.MIDDLE_FINGER_TIP],
            wrist,
        )

        ring_ext = cls._is_digit_extended(
            lms[HandLandmarkIndex.RING_FINGER_MCP],
            lms[HandLandmarkIndex.RING_FINGER_PIP],
            lms[HandLandmarkIndex.RING_FINGER_DIP],
            lms[HandLandmarkIndex.RING_FINGER_TIP],
            wrist,
        )

        pinky_ext = cls._is_digit_extended(
            lms[HandLandmarkIndex.PINKY_MCP],
            lms[HandLandmarkIndex.PINKY_PIP],
            lms[HandLandmarkIndex.PINKY_DIP],
            lms[HandLandmarkIndex.PINKY_TIP],
            wrist,
        )

        th_tip = lms[HandLandmarkIndex.THUMB_TIP]
        th_ip = lms[HandLandmarkIndex.THUMB_IP]
        idx_mcp = lms[HandLandmarkIndex.INDEX_FINGER_MCP]
        dist_th_tip_idx = math.hypot(th_tip.x - idx_mcp.x, th_tip.y - idx_mcp.y)
        dist_th_ip_idx = math.hypot(th_ip.x - idx_mcp.x, th_ip.y - idx_mcp.y)
        thumb_ext = (dist_th_tip_idx > 1.15 * dist_th_ip_idx)

        is_pinch = (normalized_pinch_dist <= pinch_threshold)

        return FingerExtensionFeatures(
            index_extended=index_ext,
            middle_extended=middle_ext,
            ring_extended=ring_ext,
            pinky_extended=pinky_ext,
            thumb_extended=thumb_ext,
            thumb_index_pinch=is_pinch,
        )


class RawGestureClassifier:
    """Classifies candidate gestures from finger extension features using priority arbitration."""

    @staticmethod
    def classify(features: FingerExtensionFeatures) -> GestureType:
        if (
            not features.index_extended and
            not features.middle_extended and
            not features.ring_extended and
            not features.pinky_extended
        ):
            return GestureType.FIST

        if features.thumb_index_pinch:
            return GestureType.PINCHING

        if (
            features.index_extended and
            features.middle_extended and
            features.ring_extended and
            features.pinky_extended
        ):
            return GestureType.OPEN_PALM

        if (
            features.index_extended and
            features.middle_extended and
            features.ring_extended and
            not features.pinky_extended
        ):
            return GestureType.THREE_FINGER

        if (
            features.index_extended and
            features.middle_extended and
            not features.ring_extended and
            not features.pinky_extended
        ):
            return GestureType.TWO_FINGER

        if (
            features.index_extended and
            not features.middle_extended and
            not features.ring_extended and
            not features.pinky_extended
        ):
            return GestureType.POINTING

        return GestureType.UNKNOWN


class TemporalStabilizer:
    """Stabilizes raw gesture candidates with activation and release debounce thresholds."""

    def __init__(self, activation_frames: int = 4, release_frames: int = 3) -> None:
        self.activation_frames = activation_frames
        self.release_frames = release_frames
        self.active_gesture: GestureType = GestureType.POINTING
        self.candidate_gesture: GestureType = GestureType.POINTING
        self.candidate_count: int = 0
        self.unknown_frame_count: int = 0

    def update(self, raw_gesture: GestureType) -> Tuple[GestureType, bool]:
        """Processes a raw classified gesture frame and returns (active_gesture, transition_committed)."""
        transition_committed = False

        if raw_gesture == GestureType.UNKNOWN:
            self.unknown_frame_count += 1
            if self.unknown_frame_count >= self.release_frames:
                if self.active_gesture != GestureType.UNKNOWN:
                    self.active_gesture = GestureType.UNKNOWN
                    transition_committed = True
            return (self.active_gesture, transition_committed)

        self.unknown_frame_count = 0

        if raw_gesture == self.candidate_gesture:
            self.candidate_count += 1
            required_frames = self.activation_frames
            if self.candidate_count >= required_frames:
                if self.candidate_gesture != self.active_gesture:
                    self.active_gesture = self.candidate_gesture
                    transition_committed = True
        else:
            self.candidate_gesture = raw_gesture
            self.candidate_count = 1

        return (self.active_gesture, transition_committed)

    def reset(self) -> None:
        """Resets temporal stabilizer state."""
        self.active_gesture = GestureType.POINTING
        self.candidate_gesture = GestureType.POINTING
        self.candidate_count = 0
        self.unknown_frame_count = 0


@dataclass
class GestureConfig:
    """Parameters for gesture recognition and interaction arbitration."""
    enabled: bool = False
    pinch_start_threshold: float = 0.35
    pinch_release_threshold: float = 0.45
    drag_threshold_px: float = 8.0
    click_cooldown_s: float = 0.20
    right_click_cooldown_s: float = 0.30
    middle_click_cooldown_s: float = 0.30
    min_pinch_duration_ms: float = 80.0
    activation_frames: int = 4
    release_frames: int = 3
    watchdog_max_drag_s: float = 30.0


class GestureEngine:
    """
    Gesture engine managing feature extraction, priority arbitration,
    temporal stabilization, and discrete mouse action dispatching.
    """

    def __init__(
        self,
        config: Optional[GestureConfig] = None,
        backend: Optional[MouseBackend] = None,
    ) -> None:
        self.config = config or GestureConfig()
        self.backend = backend or MockMouseBackend()

        self.stabilizer = TemporalStabilizer(
            activation_frames=self.config.activation_frames,
            release_frames=self.config.release_frames,
        )
        self.active_gesture: GestureType = GestureType.POINTING
        self.raw_gesture: GestureType = GestureType.POINTING
        self.latest_features: Optional[FingerExtensionFeatures] = None

        self.gesture_state: GestureState = GestureState.POINTING
        self.mouse_action: MouseActionState = MouseActionState.IDLE
        self.interaction_locked: bool = False

        self.is_pinched: bool = False
        self.normalized_pinch_dist: float = 1.0
        self.pinch_start_time: Optional[float] = None
        self.pinch_start_cursor: Optional[Tuple[int, int]] = None
        self.last_click_time: float = 0.0

        self.last_right_click_time: float = 0.0
        self.last_middle_click_time: float = 0.0

        self.is_dragging: bool = False
        self.drag_start_time: Optional[float] = None

        logger.info(
            "GestureEngine initialized. Enabled: %s, Activation Frames: %d, Pinch Start: %.2f",
            self.config.enabled,
            self.config.activation_frames,
            self.config.pinch_start_threshold,
        )

    def calculate_normalized_pinch_distance(self, hand: HandData) -> float:
        """Calculates distance-invariant normalized pinch distance."""
        if not hand or len(hand.landmarks) < 21:
            return 1.0

        it = hand.index_tip
        tt = hand.thumb_tip
        wrist = hand.wrist
        mmcp = hand.landmarks[HandLandmarkIndex.MIDDLE_FINGER_MCP]

        pinch_dist = math.sqrt((it.x - tt.x) ** 2 + (it.y - tt.y) ** 2 + (it.z - tt.z) ** 2)
        hand_scale = math.sqrt((wrist.x - mmcp.x) ** 2 + (wrist.y - mmcp.y) ** 2 + (wrist.z - mmcp.z) ** 2)
        effective_scale = max(0.02, hand_scale)

        return float(pinch_dist / effective_scale)

    def update(
        self,
        hand: Optional[HandData],
        cursor_coords: Optional[Tuple[int, int]],
    ) -> GestureType:
        """
        Processes a single frame for gesture arbitration.

        Args:
            hand: Active locked HandData or None if lost.
            cursor_coords: Screen pixel coordinates (x, y) or None if disarmed.

        Returns:
            Active GestureType.
        """
        if hand is None or cursor_coords is None:
            self._handle_hand_loss()
            return GestureType.POINTING

        now = time.perf_counter()

        norm_dist = self.calculate_normalized_pinch_distance(hand)
        self.normalized_pinch_dist = norm_dist

        if not self.is_pinched:
            if norm_dist <= self.config.pinch_start_threshold:
                self.is_pinched = True
                self.pinch_start_time = now
                self.pinch_start_cursor = cursor_coords
        else:
            if norm_dist >= self.config.pinch_release_threshold:
                self.is_pinched = False

        features = HandFeatureExtractor.extract(
            hand=hand,
            normalized_pinch_dist=norm_dist,
            pinch_threshold=self.config.pinch_start_threshold,
        )
        self.latest_features = features
        self.raw_gesture = RawGestureClassifier.classify(features)

        active_gesture, transition_committed = self.stabilizer.update(self.raw_gesture)
        self.active_gesture = active_gesture

        if transition_committed:
            self._handle_committed_transition(active_gesture, now)

        self._process_gesture_actions(active_gesture, cursor_coords, now)

        return self.active_gesture

    def _handle_committed_transition(self, gesture: GestureType, now: float) -> None:
        """Dispatches discrete actions upon stable gesture activation."""
        if gesture == GestureType.FIST:
            self.interaction_locked = True
            self.mouse_action = MouseActionState.LOCKED
            if self.is_dragging:
                logger.info("Fist detected during drag. Safety releasing mouse button.")
                if self.config.enabled:
                    self.backend.mouse_up(button="left")
                self.is_dragging = False
                self.drag_start_time = None
            logger.info("Interaction Safety Lock ACTIVE (Fist).")

        elif gesture == GestureType.OPEN_PALM:
            self.interaction_locked = False
            self.mouse_action = MouseActionState.PAUSED

        elif gesture == GestureType.TWO_FINGER:
            self.interaction_locked = False
            if (now - self.last_right_click_time) >= self.config.right_click_cooldown_s:
                if self.config.enabled:
                    self.backend.right_click()
                self.last_right_click_time = now
                self.mouse_action = MouseActionState.RIGHT_CLICKED
                logger.info("Two-Finger Right Click executed.")
            else:
                self.mouse_action = MouseActionState.IDLE

        elif gesture == GestureType.THREE_FINGER:
            self.interaction_locked = False
            cooldown = getattr(self.config, "click_cooldown_s", 0.20)
            if (now - self.last_click_time) >= cooldown:
                if self.config.enabled:
                    self.backend.click(button="left")
                self.last_click_time = now
                self.mouse_action = MouseActionState.CLICKED
                logger.info("Three-Finger Left Click executed.")
            else:
                self.mouse_action = MouseActionState.IDLE

        else:
            self.interaction_locked = False
            self.mouse_action = MouseActionState.IDLE

    def _process_gesture_actions(
        self,
        gesture: GestureType,
        cursor_coords: Tuple[int, int],
        now: float,
    ) -> None:
        """Processes continuous gestures (Pinch Click & Drag)."""
        if self.interaction_locked:
            self.gesture_state = GestureState.POINTING
            return

        if self.is_pinched:
            start_cx, start_cy = self.pinch_start_cursor or cursor_coords
            curr_cx, curr_cy = cursor_coords
            displacement = math.hypot(curr_cx - start_cx, curr_cy - start_cy)

            if displacement > self.config.drag_threshold_px:
                if not self.is_dragging:
                    if self.config.enabled:
                        success = self.backend.mouse_down(button="left")
                        if success:
                            self.is_dragging = True
                            self.drag_start_time = now
                            self.gesture_state = GestureState.DRAGGING
                            self.mouse_action = MouseActionState.BUTTON_DOWN
                        else:
                            self.gesture_state = GestureState.PINCHING
                            self.mouse_action = MouseActionState.IDLE
                    else:
                        self.is_dragging = True
                        self.drag_start_time = now
                        self.gesture_state = GestureState.DRAGGING
                        self.mouse_action = MouseActionState.BUTTON_DOWN
                else:
                    self.gesture_state = GestureState.DRAGGING
                    self.mouse_action = MouseActionState.BUTTON_DOWN

                    if self.drag_start_time and (now - self.drag_start_time) > self.config.watchdog_max_drag_s:
                        logger.warning("Drag watchdog exceeded. Triggering safety mouse_up.")
                        self.cleanup()
            else:
                if not self.is_dragging:
                    self.gesture_state = GestureState.PINCHING
                    if self.mouse_action not in (MouseActionState.RIGHT_CLICKED, MouseActionState.MIDDLE_CLICKED):
                        self.mouse_action = MouseActionState.IDLE
        else:
            if self.is_dragging:
                if self.config.enabled:
                    self.backend.mouse_up(button="left")
                self.is_dragging = False
                self.drag_start_time = None
                self.gesture_state = GestureState.RELEASED
                self.mouse_action = MouseActionState.BUTTON_UP
            elif self.pinch_start_time is not None:
                pinch_duration_ms = (now - self.pinch_start_time) * 1000.0
                if pinch_duration_ms >= self.config.min_pinch_duration_ms:
                    if (now - self.last_click_time) >= self.config.click_cooldown_s:
                        if self.config.enabled:
                            self.backend.click(button="left")
                        self.last_click_time = now
                        self.mouse_action = MouseActionState.CLICKED
                self.gesture_state = GestureState.RELEASED
            else:
                self.gesture_state = GestureState.POINTING

            self.pinch_start_time = None
            self.pinch_start_cursor = None

    def _handle_hand_loss(self) -> None:
        """Emergency release when hand tracking is lost."""
        if self.is_dragging:
            logger.info("Hand lost while dragging. Releasing mouse button.")
            if self.config.enabled:
                self.backend.mouse_up(button="left")
            self.is_dragging = False
            self.drag_start_time = None

        self.is_pinched = False
        self.pinch_start_time = None
        self.pinch_start_cursor = None
        self.interaction_locked = False
        self.stabilizer.reset()
        self.active_gesture = GestureType.POINTING
        self.gesture_state = GestureState.POINTING
        self.mouse_action = MouseActionState.IDLE

    def cleanup(self) -> None:
        """Guarantees safe release of any held mouse button during shutdown or error."""
        if self.is_dragging or (hasattr(self.backend, "is_button_down") and self.backend.is_button_down):
            logger.info("GestureEngine cleanup: releasing mouse button.")
            try:
                self.backend.mouse_up(button="left")
            except Exception as e:
                logger.error("Error during mouse_up in cleanup: %s", e)
            self.is_dragging = False
            self.drag_start_time = None

        self.is_pinched = False
        self.pinch_start_time = None
        self.pinch_start_cursor = None
        self.interaction_locked = False
        self.stabilizer.reset()
        self.active_gesture = GestureType.POINTING
        self.gesture_state = GestureState.POINTING
        self.mouse_action = MouseActionState.IDLE
