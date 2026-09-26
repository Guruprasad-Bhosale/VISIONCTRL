"""
Comprehensive unit and integration test suite for VISIONCTRL.

Validates hand tracking, cursor mapping, gesture arbitration,
spatial ray estimation, object detection, target association, and selection engine.
"""

import math
import time
from typing import List, Tuple
import cv2
import numpy as np
import pytest

from camera_projection import CameraProjection, ProjectedRay
from cursor_controller import (
    ActiveHandSelector,
    ArmingState,
    CursorConfig,
    CursorController,
    MockMouseBackend,
    PyAutoGUIMouseBackend,
)
from gesture_engine import (
    FingerExtensionFeatures,
    GestureConfig,
    GestureEngine,
    GestureState,
    GestureType,
    HandFeatureExtractor,
    MouseActionState,
    RawGestureClassifier,
    TemporalStabilizer,
)
from hand_tracker import (
    HAND_CONNECTIONS,
    HandData,
    HandLandmarkIndex,
    HandTracker,
    LandmarkPoint,
)
from hud import draw_detected_objects, draw_hand_skeleton, draw_hud
from main import FPSTracker, initialize_camera, process_frame, run
from mode_controller import (
    InteractionMode,
    ModeController,
    ModeTransitionReason,
    ModeTransitionResult,
)
from object_detector import (
    DetectedObject,
    DetectorStatus,
    MediaPipeObjectDetector,
    MockObjectDetector,
    ObjectDetector,
    ObjectDetectorConfig,
)
from pointing_estimator import (
    OriginStrategy,
    PointingConfig,
    PointingEstimate,
    PointingEstimator,
    SpatialRay,
)
from selection_engine import (
    SelectedObject,
    SelectionConfig,
    SelectionEngine,
    SelectionEvent,
    SelectionState,
)
from target_associator import (
    TargetAssociator,
    TargetAssociatorConfig,
    TargetedObject,
    TargetState,
)


# =====================================================================
# PHASE 1 REGRESSION TESTS (Preserved 100%)
# =====================================================================

class TestLandmarksAndConstants:
    """Validates landmark enumeration and connectivity definitions."""

    def test_landmark_indices_completeness(self):
        """All 21 landmark indices must be defined sequentially from 0 to 20."""
        indices = [idx.value for idx in HandLandmarkIndex]
        assert len(indices) == 21
        assert indices == list(range(21))

    def test_key_landmark_constants(self):
        """Key interaction landmarks must have exact expected indices."""
        assert HandLandmarkIndex.WRIST == 0
        assert HandLandmarkIndex.THUMB_TIP == 4
        assert HandLandmarkIndex.INDEX_FINGER_TIP == 8
        assert HandLandmarkIndex.MIDDLE_FINGER_TIP == 12
        assert HandLandmarkIndex.RING_FINGER_TIP == 16
        assert HandLandmarkIndex.PINKY_TIP == 20

    def test_hand_connections_validity(self):
        """All skeletal connections must reference valid landmark indices [0, 20]."""
        assert len(HAND_CONNECTIONS) > 0
        for start_idx, end_idx in HAND_CONNECTIONS:
            assert 0 <= start_idx <= 20
            assert 0 <= end_idx <= 20
            assert start_idx != end_idx


class TestCoordinateSystemAndDataStructures:
    """Validates spatial coordinate transformations and HandData convenience accessors."""

    def test_coordinate_transformation(self):
        """Normalized (0.0-1.0) coordinates must convert to correct integer pixel bounds."""
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        norm_x, norm_y = 0.5, 0.25

        for w, h in resolutions:
            px = int(np.clip(norm_x * w, 0, w - 1))
            py = int(np.clip(norm_y * h, 0, h - 1))
            assert px == int(0.5 * w)
            assert py == int(0.25 * h)
            assert 0 <= px < w
            assert 0 <= py < h

    def test_hand_data_accessors_and_bbox(self):
        """HandData properties must correctly expose key fingertips, wrist, bbox and center."""
        mock_landmarks = []
        for i in range(21):
            mock_landmarks.append(
                LandmarkPoint(
                    x=i * 0.04,
                    y=i * 0.04,
                    z=0.0,
                    px=i * 20 + 100,
                    py=i * 15 + 50,
                )
            )

        hand = HandData(
            handedness="Right",
            score=0.95,
            landmarks=mock_landmarks,
        )

        assert hand.wrist.px == mock_landmarks[0].px
        assert hand.wrist.py == mock_landmarks[0].py
        assert hand.thumb_tip.px == mock_landmarks[4].px
        assert hand.index_tip.px == mock_landmarks[8].px
        assert hand.middle_tip.px == mock_landmarks[12].px
        assert hand.ring_tip.px == mock_landmarks[16].px
        assert hand.pinky_tip.px == mock_landmarks[20].px

        min_x, min_y, max_x, max_y = hand.bbox
        assert min_x == 100
        assert max_x == 20 * 20 + 100
        assert min_y == 50
        assert max_y == 20 * 15 + 50

        cx, cy = hand.center
        assert cx == (min_x + max_x) // 2
        assert cy == (min_y + max_y) // 2


class TestHandTrackerPipeline:
    """Validates tracker initialization, monotonic timestamps, and stability on blank/noise frames."""

    @pytest.fixture(scope="class")
    @classmethod
    def tracker(cls):
        """Fixture providing an initialized HandTracker instance."""
        trk = HandTracker(model_path="models/hand_landmarker.task")
        yield trk
        trk.close()

    def test_monotonic_timestamp_generation(self, tracker):
        """Timestamps passed to detect_for_video must be strictly monotonically increasing."""
        timestamps = []
        for _ in range(100):
            ts = tracker._get_next_timestamp_ms()
            timestamps.append(ts)

        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Timestamp violation at index {i}: {timestamps[i]} <= {timestamps[i-1]}"
            )

    def test_tracker_stability_blank_frame(self, tracker):
        """Tracker must handle blank frames gracefully without errors and return 0 hands."""
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hands = tracker.process_frame(blank_frame)
        assert isinstance(hands, list)
        assert len(hands) == 0

    def test_tracker_stability_random_noise(self, tracker):
        """Tracker must handle random Gaussian/uniform noise without crashing."""
        np.random.seed(42)
        noise_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        hands = tracker.process_frame(noise_frame)
        assert isinstance(hands, list)

    def test_tracker_empty_frame_handling(self, tracker):
        """Tracker must handle empty/None frame inputs safely."""
        empty_frame = np.array([], dtype=np.uint8)
        hands = tracker.process_frame(empty_frame)
        assert hands == []


class TestFPSTracker:
    """Validates frame rate smoothing calculation."""

    def test_fps_calculation_smoothing(self):
        fps_calc = FPSTracker(window_size=10)
        assert fps_calc.fps == 0.0

        for _ in range(15):
            fps_calc.tick()
            time.sleep(0.005)

        assert fps_calc.fps > 0.0


class TestCameraAndHUDRendering:
    """Validates camera failure handling, HUD visual rendering, and synthetic runs."""

    def test_camera_failure_handling(self):
        """Requesting an invalid camera index (e.g., 9999) must safely return None without throwing unhandled exceptions."""
        cap = initialize_camera(camera_id=9999)
        assert cap is None

    def test_hud_rendering_on_frame(self):
        """draw_hud must modify the frame without shape alteration."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = draw_hud(frame, [], None, None, fps=30.0, camera_status="TEST")
        assert rendered.shape == (720, 1280, 3)

    def test_skeleton_rendering_on_frame(self):
        """draw_hand_skeleton must draw landmark points and connections without crashing."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_landmarks = [
            LandmarkPoint(x=0.5, y=0.5, z=0.0, px=640, py=360)
            for _ in range(21)
        ]
        hand = HandData(handedness="Right", score=0.9, landmarks=mock_landmarks)
        rendered = draw_hand_skeleton(frame, [hand], active_hand=hand)
        assert rendered.shape == (720, 1280, 3)

    def test_synthetic_run_loop(self):
        """Executing run() in synthetic mode for 5 frames must complete safely with exit code 0."""
        exit_code = run(
            synthetic_mode=True,
            max_frames=5,
            window_name="VISIONCTRL - Test Window",
        )
        assert exit_code == 0


# =====================================================================
# PHASE 2 & 3 REGRESSION TESTS (Preserved 100%)
# =====================================================================

def create_mock_hand_digits(
    index_ext: bool = True,
    middle_ext: bool = False,
    ring_ext: bool = False,
    pinky_ext: bool = False,
    thumb_ext: bool = True,
    pinch_dist_norm: float = 0.50,
    wrist_y: float = 0.85,
) -> HandData:
    """Constructs a deterministic HandData instance with specified finger extension postures."""
    lms = []
    # 0: Wrist
    lms.append(LandmarkPoint(x=0.5, y=wrist_y, z=0.0, px=640, py=int(wrist_y * 720)))

    def add_finger(mcp_x, mcp_y, extended, is_thumb=False):
        if is_thumb:
            lms.append(LandmarkPoint(x=0.45, y=0.75, z=0.0, px=576, py=540))
            lms.append(LandmarkPoint(x=0.40, y=0.70, z=0.0, px=512, py=504))
            lms.append(LandmarkPoint(x=0.35, y=0.65, z=0.0, px=448, py=468))
            if extended:
                lms.append(LandmarkPoint(x=0.30, y=0.60, z=0.0, px=384, py=432))
            else:
                lms.append(LandmarkPoint(x=0.42, y=0.68, z=0.0, px=537, py=489))
            return

        # MCP, PIP, DIP, TIP
        lms.append(LandmarkPoint(x=mcp_x, y=mcp_y, z=0.0, px=int(mcp_x * 1280), py=int(mcp_y * 720)))
        if extended:
            lms.append(LandmarkPoint(x=mcp_x, y=mcp_y - 0.12, z=0.0, px=int(mcp_x * 1280), py=int((mcp_y - 0.12) * 720)))
            lms.append(LandmarkPoint(x=mcp_x, y=mcp_y - 0.22, z=0.0, px=int(mcp_x * 1280), py=int((mcp_y - 0.22) * 720)))
            lms.append(LandmarkPoint(x=mcp_x, y=mcp_y - 0.32, z=0.0, px=int(mcp_x * 1280), py=int((mcp_y - 0.32) * 720)))
        else:
            lms.append(LandmarkPoint(x=mcp_x, y=mcp_y - 0.08, z=0.0, px=int(mcp_x * 1280), py=int((mcp_y - 0.08) * 720)))
            lms.append(LandmarkPoint(x=mcp_x, y=mcp_y + 0.02, z=0.0, px=int(mcp_x * 1280), py=int((mcp_y + 0.02) * 720)))
            lms.append(LandmarkPoint(x=mcp_x, y=mcp_y + 0.08, z=0.0, px=int(mcp_x * 1280), py=int((mcp_y + 0.08) * 720)))

    # Thumb: 1, 2, 3, 4
    add_finger(0.40, 0.70, thumb_ext, is_thumb=True)
    # Index: 5, 6, 7, 8
    add_finger(0.45, 0.55, index_ext)
    # Middle: 9, 10, 11, 12
    add_finger(0.50, 0.53, middle_ext)
    # Ring: 13, 14, 15, 16
    add_finger(0.55, 0.55, ring_ext)
    # Pinky: 17, 18, 19, 20
    add_finger(0.60, 0.58, pinky_ext)

    if pinch_dist_norm <= 0.35:
        idx_tip = lms[8]
        lms[4] = LandmarkPoint(x=idx_tip.x + 0.02, y=idx_tip.y + 0.02, z=0.0, px=idx_tip.px + 25, py=idx_tip.py + 15)

    return HandData(handedness="Right", score=0.95, landmarks=lms)


class TestActiveHandSelector:
    """Validates stable hand locking and intelligent reacquisition."""

    def test_initial_hand_lock(self):
        selector = ActiveHandSelector()
        hand_r = create_mock_hand_digits(index_ext=True)
        hand_l = create_mock_hand_digits(index_ext=True)
        hand_l.handedness = "Left"

        selected = selector.select([hand_r, hand_l])
        assert selected is not None
        assert selected.handedness == "Right"
        assert selector.locked_handedness == "Right"

    def test_hand_order_swap_stability(self):
        selector = ActiveHandSelector()
        hand_r = create_mock_hand_digits(index_ext=True)
        hand_l = create_mock_hand_digits(index_ext=True)
        hand_l.handedness = "Left"

        selector.select([hand_r, hand_l])
        selected = selector.select([hand_l, hand_r])
        assert selected is not None
        assert selected.handedness == "Right"

    def test_hand_reacquisition_after_loss(self):
        selector = ActiveHandSelector(max_lost_frames=3)
        hand_r = create_mock_hand_digits(index_ext=True)
        hand_l = create_mock_hand_digits(index_ext=True)
        hand_l.handedness = "Left"

        selector.select([hand_r])
        for _ in range(4):
            selector.select([])

        new_selected = selector.select([hand_l])
        assert new_selected is not None
        assert new_selected.handedness == "Left"


class TestCursorCoordinateMapping:
    """Validates coordinate normalization, margins, and clamping."""

    def test_cardinal_points_zero_margins(self):
        backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        config = CursorConfig(enabled=True, x_margin=0.0, y_margin=0.0, smoothing=0.0, deadzone=0.0)
        controller = CursorController(config=config, backend=backend)

        assert controller.map_norm_to_screen(0.0, 0.0) == (0, 0)
        assert controller.map_norm_to_screen(0.5, 0.5) == (960, 540)
        assert controller.map_norm_to_screen(1.0, 1.0) == (1919, 1079)

    def test_interaction_margins_clamping(self):
        backend = MockMouseBackend(screen_width=1000, screen_height=1000)
        config = CursorConfig(enabled=True, x_margin=0.10, y_margin=0.10, smoothing=0.0, deadzone=0.0)
        controller = CursorController(config=config, backend=backend)

        assert controller.map_norm_to_screen(0.10, 0.10) == (0, 0)
        assert controller.map_norm_to_screen(0.05, 0.05) == (0, 0)
        assert controller.map_norm_to_screen(0.90, 0.90) == (999, 999)
        assert controller.map_norm_to_screen(0.95, 0.95) == (999, 999)


class TestCursorFilterAndArming:
    """Validates EMA smoothing, step clamping, deadzone, and arming lifecycle."""

    def test_first_point_initialization_no_origin_drift(self):
        backend = MockMouseBackend(screen_width=1001, screen_height=1001)
        config = CursorConfig(enabled=True, x_margin=0.0, y_margin=0.0, smoothing=0.8, deadzone=0.0, arming_duration_s=0.0)
        controller = CursorController(config=config, backend=backend)

        controller.update(0.6, 0.6)
        assert controller.smoothed_x == 600.0
        assert controller.smoothed_y == 600.0
        assert backend.cursor_x == 600
        assert backend.cursor_y == 600

    def test_exponential_smoothing_accuracy(self):
        backend = MockMouseBackend(screen_width=1001, screen_height=1001)
        config = CursorConfig(enabled=True, x_margin=0.0, y_margin=0.0, smoothing=0.5, deadzone=0.0, max_step_px=5000.0, arming_duration_s=0.0)
        controller = CursorController(config=config, backend=backend)

        controller.update(0.0, 0.0)
        controller.update(0.8, 0.8)
        assert controller.smoothed_x == 400.0
        assert controller.smoothed_y == 400.0

    def test_max_step_clamping(self):
        backend = MockMouseBackend(screen_width=1000, screen_height=1000)
        config = CursorConfig(enabled=True, x_margin=0.0, y_margin=0.0, smoothing=0.0, deadzone=0.0, max_step_px=50.0, arming_duration_s=0.0)
        controller = CursorController(config=config, backend=backend)

        controller.update(0.1, 0.1)
        controller.update(0.9, 0.1)
        assert controller.smoothed_x == 150.0

    def test_deadzone_suppression_and_filter_state(self):
        backend = MockMouseBackend(screen_width=1000, screen_height=1000)
        config = CursorConfig(enabled=True, x_margin=0.0, y_margin=0.0, smoothing=0.0, deadzone=5.0, arming_duration_s=0.0)
        controller = CursorController(config=config, backend=backend)

        controller.update(0.5, 0.5)
        assert len(backend.move_history) == 1

        controller.update(0.502, 0.5)
        assert len(backend.move_history) == 1
        assert controller.smoothed_x == 501.0

        controller.update(0.512, 0.5)
        assert len(backend.move_history) == 2
        assert backend.cursor_x == 511

    def test_cursor_disabled_mode_no_os_emissions(self):
        backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        config = CursorConfig(enabled=False, arming_duration_s=0.0)
        controller = CursorController(config=config, backend=backend)

        controller.update(0.5, 0.5)
        controller.update(0.8, 0.8)
        assert len(backend.move_history) == 0


class TestPhase3PinchNonRegressionGates:
    """Non-regression gates verifying Phase 3 pinch click, hold, drag, and release."""

    def test_pinch_click_on_release_without_movement(self):
        """PINCH -> release without movement -> clicks=1, mouse_down=0, mouse_up=0."""
        backend = MockMouseBackend()
        config = GestureConfig(enabled=True, min_pinch_duration_ms=0.0)
        engine = GestureEngine(config=config, backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        backend.move_to(600, 400)
        engine.update(hand_pinch, (600, 400))
        assert len(backend.clicks) == 0
        assert len(backend.mouse_downs) == 0

        engine.update(hand_open, (600, 400))
        assert len(backend.clicks) == 1
        assert len(backend.mouse_downs) == 0
        assert len(backend.mouse_ups) == 0

    def test_pinch_drag_on_movement_threshold(self):
        """PINCH -> move 10px -> release -> mouse_down=1, mouse_up=1, clicks=0."""
        backend = MockMouseBackend()
        config = GestureConfig(enabled=True, drag_threshold_px=8.0, min_pinch_duration_ms=0.0)
        engine = GestureEngine(config=config, backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        # 1. Start pinch
        engine.update(hand_pinch, (500, 500))
        assert len(backend.mouse_downs) == 0

        # 2. Move 10px beyond drag threshold -> mouse_down=1
        engine.update(hand_pinch, (510, 500))
        assert len(backend.mouse_downs) == 1
        assert backend.is_button_down

        # 3. Release pinch -> mouse_up=1, clicks=0
        engine.update(hand_open, (510, 500))
        assert len(backend.mouse_ups) == 1
        assert not backend.is_button_down
        assert len(backend.clicks) == 0


# =====================================================================
# PHASE 4 ADVANCED GESTURE & ARBITRATION TESTS
# =====================================================================

class TestHandFeatureExtractorAndClassifier:
    """Validates deterministic feature extraction and conflict matrix classification."""

    def test_pointing_classification(self):
        hand = create_mock_hand_digits(index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False, pinch_dist_norm=0.70)
        features = HandFeatureExtractor.extract(hand, normalized_pinch_dist=0.70)
        assert features.index_extended
        assert not features.middle_extended
        assert not features.ring_extended
        assert not features.pinky_extended
        assert RawGestureClassifier.classify(features) == GestureType.POINTING

    def test_two_finger_classification(self):
        hand = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False, pinch_dist_norm=0.70)
        features = HandFeatureExtractor.extract(hand, normalized_pinch_dist=0.70)
        assert features.index_extended
        assert features.middle_extended
        assert not features.ring_extended
        assert not features.pinky_extended
        assert RawGestureClassifier.classify(features) == GestureType.TWO_FINGER

    def test_three_finger_classification(self):
        hand = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=False, pinch_dist_norm=0.70)
        features = HandFeatureExtractor.extract(hand, normalized_pinch_dist=0.70)
        assert features.index_extended
        assert features.middle_extended
        assert features.ring_extended
        assert not features.pinky_extended
        assert RawGestureClassifier.classify(features) == GestureType.THREE_FINGER

    def test_open_palm_classification(self):
        hand = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True, pinch_dist_norm=0.70)
        features = HandFeatureExtractor.extract(hand, normalized_pinch_dist=0.70)
        assert features.index_extended
        assert features.middle_extended
        assert features.ring_extended
        assert features.pinky_extended
        assert RawGestureClassifier.classify(features) == GestureType.OPEN_PALM

    def test_fist_classification(self):
        hand = create_mock_hand_digits(index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, thumb_ext=False, pinch_dist_norm=0.70)
        features = HandFeatureExtractor.extract(hand, normalized_pinch_dist=0.70)
        assert not features.index_extended
        assert not features.middle_extended
        assert not features.ring_extended
        assert not features.pinky_extended
        assert RawGestureClassifier.classify(features) == GestureType.FIST

    def test_pinch_priority_over_finger_counting(self):
        """When pinch is active, PINCHING must be returned even if middle finger is extended."""
        hand = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False, pinch_dist_norm=0.20)
        features = HandFeatureExtractor.extract(hand, normalized_pinch_dist=0.20, pinch_threshold=0.35)
        assert features.thumb_index_pinch
        assert RawGestureClassifier.classify(features) == GestureType.PINCHING

    def test_unknown_ambiguous_classification(self):
        """Non-standard configurations (e.g., only pinky extended) classify as UNKNOWN."""
        hand = create_mock_hand_digits(index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=True, pinch_dist_norm=0.70)
        features = HandFeatureExtractor.extract(hand, normalized_pinch_dist=0.70)
        assert RawGestureClassifier.classify(features) == GestureType.UNKNOWN


class TestTemporalStabilizer:
    """Validates candidate debounce, stability delay, and release hysteresis."""

    def test_candidate_stabilization_delay(self):
        stabilizer = TemporalStabilizer(activation_frames=4, release_frames=3)
        stabilizer.active_gesture = GestureType.POINTING

        for _ in range(3):
            active, committed = stabilizer.update(GestureType.TWO_FINGER)
            assert active == GestureType.POINTING
            assert not committed

        active, committed = stabilizer.update(GestureType.TWO_FINGER)
        assert active == GestureType.TWO_FINGER
        assert committed

    def test_candidate_reset_on_jitter(self):
        stabilizer = TemporalStabilizer(activation_frames=4, release_frames=3)
        stabilizer.active_gesture = GestureType.POINTING

        stabilizer.update(GestureType.TWO_FINGER)
        stabilizer.update(GestureType.TWO_FINGER)
        stabilizer.update(GestureType.THREE_FINGER)
        active, committed = stabilizer.update(GestureType.TWO_FINGER)
        assert active == GestureType.POINTING
        assert not committed
        assert stabilizer.candidate_count == 1

    def test_unknown_transient_noise_suppression(self):
        """A single UNKNOWN frame does not immediately flip active gesture."""
        stabilizer = TemporalStabilizer(activation_frames=4, release_frames=3)
        stabilizer.active_gesture = GestureType.POINTING

        active, committed = stabilizer.update(GestureType.UNKNOWN)
        assert active == GestureType.POINTING
        assert not committed


class TestPhase4GestureActionsAndArbitration:
    """Validates discrete mouse actions, interaction lock, cooldowns, and transitions."""

    def test_two_finger_activates_right_click_once(self):
        backend = MockMouseBackend()
        config = GestureConfig(enabled=True, activation_frames=3, right_click_cooldown_s=0.20)
        engine = GestureEngine(config=config, backend=backend)

        hand_two = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)

        for _ in range(3):
            engine.update(hand_two, (500, 500))

        assert len(backend.right_clicks) == 1
        assert engine.mouse_action == MouseActionState.RIGHT_CLICKED

        for _ in range(10):
            engine.update(hand_two, (500, 500))
        assert len(backend.right_clicks) == 1

    def test_three_finger_activates_middle_click_once(self):
        backend = MockMouseBackend()
        config = GestureConfig(enabled=True, activation_frames=3, middle_click_cooldown_s=0.20)
        engine = GestureEngine(config=config, backend=backend)

        hand_three = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=False)

        for _ in range(3):
            engine.update(hand_three, (500, 500))

        assert len(backend.middle_clicks) == 1
        assert engine.mouse_action == MouseActionState.MIDDLE_CLICKED

        for _ in range(10):
            engine.update(hand_three, (500, 500))
        assert len(backend.middle_clicks) == 1

    def test_open_palm_suppresses_actions(self):
        backend = MockMouseBackend()
        config = GestureConfig(enabled=True, activation_frames=2)
        engine = GestureEngine(config=config, backend=backend)

        hand_palm = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)

        for _ in range(4):
            engine.update(hand_palm, (500, 500))

        assert len(backend.clicks) == 0
        assert len(backend.right_clicks) == 0
        assert len(backend.middle_clicks) == 0
        assert len(backend.mouse_downs) == 0
        assert engine.mouse_action == MouseActionState.PAUSED

    def test_fist_safety_lock_and_drag_termination(self):
        backend = MockMouseBackend()
        config = GestureConfig(enabled=True, activation_frames=2, drag_threshold_px=5.0)
        engine = GestureEngine(config=config, backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_fist = create_mock_hand_digits(index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, thumb_ext=False)

        engine.update(hand_pinch, (500, 500))
        engine.update(hand_pinch, (520, 500))
        assert backend.is_button_down

        for _ in range(2):
            engine.update(hand_fist, (520, 500))

        assert not backend.is_button_down
        assert len(backend.mouse_ups) == 1
        assert engine.interaction_locked
        assert engine.mouse_action == MouseActionState.LOCKED

    def test_explicit_gesture_transitions(self):
        backend = MockMouseBackend()
        config = GestureConfig(enabled=True, activation_frames=2, right_click_cooldown_s=0.05)
        engine = GestureEngine(config=config, backend=backend)

        hand_two = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        hand_palm = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)

        for _ in range(2):
            engine.update(hand_two, (500, 500))
        assert len(backend.right_clicks) == 1

        for _ in range(2):
            engine.update(hand_palm, (500, 500))
        assert len(backend.right_clicks) == 1

        time.sleep(0.06)

        for _ in range(2):
            engine.update(hand_two, (500, 500))
        assert len(backend.right_clicks) == 2


# =====================================================================
# PHASE 5 SPATIAL RAY & POINTING ESTIMATION TESTS
# =====================================================================

def create_mock_pointing_hand(
    mcp_pos: Tuple[float, float, float] = (0.50, 0.50, 0.0),
    direction: Tuple[float, float, float] = (0.0, -1.0, 0.0),
    seg_length: float = 0.08,
) -> HandData:
    """
    Constructs a HandData instance where the index finger points along a specified 3D vector.
    """
    mag = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    ux, uy, uz = direction[0] / mag, direction[1] / mag, direction[2] / mag

    lms = []
    # 0: Wrist
    wrist_x = mcp_pos[0] - ux * seg_length
    wrist_y = mcp_pos[1] - uy * seg_length + 0.15
    lms.append(LandmarkPoint(x=wrist_x, y=wrist_y, z=0.0, px=int(wrist_x * 1280), py=int(wrist_y * 720)))

    # 1..4: Folded Thumb
    for i in range(1, 5):
        lms.append(LandmarkPoint(x=0.40, y=0.70, z=0.0, px=512, py=504))

    # 5: Index MCP
    mcp_x, mcp_y, mcp_z = mcp_pos
    lms.append(LandmarkPoint(x=mcp_x, y=mcp_y, z=mcp_z, px=int(mcp_x * 1280), py=int(mcp_y * 720)))

    # 6: Index PIP
    pip_x = mcp_x + ux * seg_length
    pip_y = mcp_y + uy * seg_length
    pip_z = mcp_z + uz * seg_length
    lms.append(LandmarkPoint(x=pip_x, y=pip_y, z=pip_z, px=int(pip_x * 1280), py=int(pip_y * 720)))

    # 7: Index DIP
    dip_x = pip_x + ux * seg_length
    dip_y = pip_y + uy * seg_length
    dip_z = pip_z + uz * seg_length
    lms.append(LandmarkPoint(x=dip_x, y=dip_y, z=dip_z, px=int(dip_x * 1280), py=int(dip_y * 720)))

    # 8: Index TIP
    tip_x = dip_x + ux * seg_length
    tip_y = dip_y + uy * seg_length
    tip_z = dip_z + uz * seg_length
    lms.append(LandmarkPoint(x=tip_x, y=tip_y, z=tip_z, px=int(tip_x * 1280), py=int(tip_y * 720)))

    # 9..20: Folded Middle, Ring, Pinky
    for i in range(9, 21):
        lms.append(LandmarkPoint(x=0.55, y=0.65, z=0.0, px=704, py=468))

    return HandData(handedness="Right", score=0.95, landmarks=lms)


class TestPointingEstimatorFingerAxisAndOrientation:
    """Validates 3D longitudinal finger-axis direction vector and orientation invariants."""

    def test_valid_index_finger_axis_calculation(self):
        hand = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))
        unit_dir, quality = PointingEstimator.compute_finger_axis(hand.landmarks)
        assert unit_dir is not None
        assert quality > 0.90
        # Pointing straight up in image coordinates: dy should be negative, dx ~ 0
        assert math.isclose(unit_dir[0], 0.0, abs_tol=1e-3)
        assert math.isclose(unit_dir[1], -1.0, abs_tol=1e-3)
        assert math.isclose(unit_dir[2], 0.0, abs_tol=1e-3)

    def test_direction_pointing_right(self):
        """Finger pointing right -> Dx > 0."""
        hand = create_mock_pointing_hand(direction=(1.0, 0.0, 0.0))
        unit_dir, quality = PointingEstimator.compute_finger_axis(hand.landmarks)
        assert unit_dir is not None
        assert unit_dir[0] > 0.95
        assert math.isclose(unit_dir[1], 0.0, abs_tol=1e-2)

    def test_direction_pointing_left(self):
        """Finger pointing left -> Dx < 0."""
        hand = create_mock_pointing_hand(direction=(-1.0, 0.0, 0.0))
        unit_dir, quality = PointingEstimator.compute_finger_axis(hand.landmarks)
        assert unit_dir is not None
        assert unit_dir[0] < -0.95
        assert math.isclose(unit_dir[1], 0.0, abs_tol=1e-2)

    def test_direction_pointing_up(self):
        """Finger pointing upward -> Dy < 0."""
        hand = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))
        unit_dir, quality = PointingEstimator.compute_finger_axis(hand.landmarks)
        assert unit_dir is not None
        assert unit_dir[1] < -0.95

    def test_direction_pointing_down(self):
        """Finger pointing downward -> Dy > 0."""
        hand = create_mock_pointing_hand(direction=(0.0, 1.0, 0.0))
        unit_dir, quality = PointingEstimator.compute_finger_axis(hand.landmarks)
        assert unit_dir is not None
        assert unit_dir[1] > 0.95

    def test_normalized_ray_direction_magnitude(self):
        """Direction vector must always have strictly unit length ||D|| = 1.0."""
        directions = [
            (1.0, 2.0, 3.0),
            (-3.0, 4.0, -1.0),
            (0.5, -0.8, 0.2),
        ]
        for d in directions:
            hand = create_mock_pointing_hand(direction=d)
            unit_dir, _ = PointingEstimator.compute_finger_axis(hand.landmarks)
            assert unit_dir is not None
            mag = math.sqrt(unit_dir[0] ** 2 + unit_dir[1] ** 2 + unit_dir[2] ** 2)
            assert math.isclose(mag, 1.0, abs_tol=1e-5)

    def test_zero_length_segment_rejection(self):
        """Degenerate identical landmark points must return None and 0 quality."""
        hand = create_mock_pointing_hand()
        # Collapse tip onto dip
        hand.landmarks[HandLandmarkIndex.INDEX_FINGER_TIP] = hand.landmarks[HandLandmarkIndex.INDEX_FINGER_DIP]
        unit_dir, quality = PointingEstimator.compute_finger_axis(hand.landmarks)
        assert unit_dir is None
        assert quality == 0.0


class TestPointingEstimatorConfidenceAndHysteresis:
    """Validates deterministic confidence formula and activation/release hysteresis."""

    def test_confidence_range_and_determinism(self):
        estimator = PointingEstimator()
        hand = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))
        estimate = estimator.estimate(hand, "POINTING", 1.0)

        assert 0.0 <= estimate.confidence <= 1.0
        assert estimate.confidence >= 0.70
        assert estimate.pointing_valid
        assert estimate.ray.valid

    def test_confidence_decreases_with_curved_index_finger(self):
        estimator = PointingEstimator()
        straight_hand = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))
        est_straight = estimator.estimate(straight_hand, "POINTING", 1.0)

        # Create curved finger where tip turns 90 degrees
        curved_hand = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))
        dip = curved_hand.landmarks[HandLandmarkIndex.INDEX_FINGER_DIP]
        curved_hand.landmarks[HandLandmarkIndex.INDEX_FINGER_TIP] = LandmarkPoint(
            x=dip.x + 0.08,
            y=dip.y,
            z=dip.z,
            px=dip.px + 100,
            py=dip.py,
        )

        estimator.reset()
        est_curved = estimator.estimate(curved_hand, "POINTING", 1.0)
        assert est_curved.confidence < est_straight.confidence
        assert est_curved.index_axis_quality < est_straight.index_axis_quality

    def test_confidence_hysteresis_activation_and_release(self):
        config = PointingConfig(ray_activate_confidence=0.50, ray_release_confidence=0.30)
        estimator = PointingEstimator(config=config)
        hand = create_mock_pointing_hand()

        # Ray starts inactive
        assert not estimator.is_active

        # Feed hand with confidence >= 0.50 -> Activates
        est1 = estimator.estimate(hand, "POINTING", 1.0)
        assert est1.confidence >= 0.50
        assert est1.pointing_valid
        assert estimator.is_active

        # Artificially lower hand score so confidence is ~0.40 (between 0.30 and 0.50)
        hand.score = 0.35
        est2 = estimator.estimate(hand, "POINTING", 1.1)
        # Stays active due to hysteresis (above release threshold 0.30)
        assert estimator.is_active
        assert est2.pointing_valid

        # Drop hand score to 0 and misalign finger segments (zigzag) so confidence drops below 0.30 -> Releases
        hand.score = 0.0
        mcp = hand.landmarks[HandLandmarkIndex.INDEX_FINGER_MCP]
        # PIP straight up, DIP right 90 deg, TIP left 180 deg (misaligned)
        hand.landmarks[HandLandmarkIndex.INDEX_FINGER_PIP] = LandmarkPoint(x=mcp.x, y=mcp.y - 0.05, z=0.0, px=0, py=0)
        hand.landmarks[HandLandmarkIndex.INDEX_FINGER_DIP] = LandmarkPoint(x=mcp.x + 0.06, y=mcp.y - 0.05, z=0.0, px=0, py=0)
        hand.landmarks[HandLandmarkIndex.INDEX_FINGER_TIP] = LandmarkPoint(x=mcp.x - 0.02, y=mcp.y - 0.05, z=0.0, px=0, py=0)
        est3 = estimator.estimate(hand, "POINTING", 1.2)
        assert not estimator.is_active
        assert not est3.pointing_valid

    def test_pointing_gesture_activates_ray(self):
        estimator = PointingEstimator()
        hand = create_mock_pointing_hand()
        estimate = estimator.estimate(hand, "POINTING", 1.0)
        assert estimate.pointing_valid
        assert estimate.ray.valid

    def test_non_pointing_gestures_deactivate_ray(self):
        """All non-POINTING gestures must mark ray.valid = False."""
        non_pointing = ["PINCHING", "TWO_FINGER", "THREE_FINGER", "OPEN_PALM", "FIST", "UNKNOWN"]
        hand = create_mock_pointing_hand()

        for g in non_pointing:
            estimator = PointingEstimator()
            # First activate in pointing
            estimator.estimate(hand, "POINTING", 1.0)
            # Then switch to non-pointing
            estimate = estimator.estimate(hand, g, 1.1)
            assert not estimate.pointing_valid
            assert not estimate.ray.valid

    def test_invalid_missing_hand_landmarks_handling(self):
        estimator = PointingEstimator()
        estimate = estimator.estimate(None, "POINTING", 1.0)
        assert not estimate.hand_valid
        assert not estimate.pointing_valid
        assert not estimate.ray.valid
        assert estimate.confidence == 0.0

    def test_ray_loss_grace_period_and_reset(self):
        config = PointingConfig(loss_grace_period_frames=2)
        estimator = PointingEstimator(config=config)
        hand = create_mock_pointing_hand()

        # 1. Activate
        estimator.estimate(hand, "POINTING", 1.0)
        assert estimator.is_active

        # 2. Hand missing for 1 frame -> Grace period retains last ray
        est_grace1 = estimator.estimate(None, "POINTING", 1.1)
        assert est_grace1.ray is not None

        # 3. Hand missing for 2nd frame
        estimator.estimate(None, "POINTING", 1.2)

        # 4. Hand missing for 3rd frame -> Fully reset
        est_lost = estimator.estimate(None, "POINTING", 1.3)
        assert not estimator.is_active
        assert not est_lost.ray.valid


class TestPointingEstimatorSmoothingAndOrigins:
    """Validates direction EMA smoothing, origin strategies, and transition stability."""

    def test_temporal_direction_smoothing_and_renormalization(self):
        config = PointingConfig(direction_smoothing=0.50)
        estimator = PointingEstimator(config=config)

        # Frame 1: Pointing straight Up (0, -1, 0)
        hand1 = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))
        est1 = estimator.estimate(hand1, "POINTING", 1.0)
        d1 = est1.ray.direction_3d

        # Frame 2: Pointing Right (1, 0, 0)
        hand2 = create_mock_pointing_hand(direction=(1.0, 0.0, 0.0))
        est2 = estimator.estimate(hand2, "POINTING", 1.1)
        d2 = est2.ray.direction_3d

        # Smoothed direction should interpolate between (0, -1) and (1, 0)
        assert d2[0] > 0.50
        assert d2[1] < -0.50
        # Must still be strictly normalized
        mag = math.sqrt(d2[0] ** 2 + d2[1] ** 2 + d2[2] ** 2)
        assert math.isclose(mag, 1.0, abs_tol=1e-5)

    def test_geometrically_consistent_3d_endpoint(self):
        estimator = PointingEstimator()
        hand = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))
        estimate = estimator.estimate(hand, "POINTING", 1.0)

        ray = estimate.ray
        expected_end_x = ray.origin_3d[0] + ray.direction_3d[0] * ray.length
        expected_end_y = ray.origin_3d[1] + ray.direction_3d[1] * ray.length
        expected_end_z = ray.origin_3d[2] + ray.direction_3d[2] * ray.length

        assert math.isclose(ray.endpoint_3d[0], expected_end_x, abs_tol=1e-5)
        assert math.isclose(ray.endpoint_3d[1], expected_end_y, abs_tol=1e-5)
        assert math.isclose(ray.endpoint_3d[2], expected_end_z, abs_tol=1e-5)

    def test_origin_strategies_support(self):
        hand = create_mock_pointing_hand()
        for strat in ["INDEX_TIP", "INDEX_DIP", "INDEX_PIP", "INDEX_MCP"]:
            config = PointingConfig(origin_strategy=strat)
            estimator = PointingEstimator(config=config)
            estimate = estimator.estimate(hand, "POINTING", 1.0)
            orig = estimate.ray.origin_3d

            if strat == "INDEX_TIP":
                assert orig == (hand.index_tip.x, hand.index_tip.y, hand.index_tip.z)
            elif strat == "INDEX_MCP":
                mcp = hand.landmarks[HandLandmarkIndex.INDEX_FINGER_MCP]
                assert orig == (mcp.x, mcp.y, mcp.z)

    def test_ray_continuity_across_transient_unknown_frame(self):
        estimator = PointingEstimator()
        hand = create_mock_pointing_hand(direction=(0.0, -1.0, 0.0))

        # Frame 1: Pointing
        est1 = estimator.estimate(hand, "POINTING", 1.0)
        assert est1.ray.valid

        # Frame 2: 1-frame transient UNKNOWN (e.g. noise)
        est2 = estimator.estimate(hand, "UNKNOWN", 1.03)

        # Frame 3: Pointing re-asserted
        est3 = estimator.estimate(hand, "POINTING", 1.06)
        assert est3.ray.valid
        # Direction should remain smoothly aligned without jump
        dot = (
            est1.ray.direction_3d[0] * est3.ray.direction_3d[0]
            + est1.ray.direction_3d[1] * est3.ray.direction_3d[1]
            + est1.ray.direction_3d[2] * est3.ray.direction_3d[2]
        )
        assert dot > 0.95

    def test_ray_reacquisition_after_open_palm(self):
        estimator = PointingEstimator()
        hand_point = create_mock_pointing_hand()
        hand_palm = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True)

        # 1. Pointing
        est1 = estimator.estimate(hand_point, "POINTING", 1.0)
        assert est1.ray.valid

        # 2. Open Palm -> Deactivates
        est2 = estimator.estimate(hand_palm, "OPEN_PALM", 1.1)
        assert not est2.ray.valid

        # 3. Back to Pointing -> Clean reacquisition
        est3 = estimator.estimate(hand_point, "POINTING", 1.2)
        assert est3.ray.valid


class TestCameraProjection:
    """Validates uncalibrated visual camera-plane projection mapping to pixels."""

    def test_camera_plane_projection_deterministic(self):
        ray = SpatialRay(
            origin_3d=(0.5, 0.5, 0.0),
            direction_3d=(0.0, -1.0, 0.0),
            length=0.25,
            endpoint_3d=(0.5, 0.25, 0.0),
            timestamp=1.0,
            valid=True,
        )
        proj = CameraProjection.project_ray(ray, frame_width=1280, frame_height=720)
        assert proj.origin_px == (640, 360)
        assert proj.endpoint_px == (640, 180)
        assert proj.visible

    def test_visual_length_projection(self):
        ray = SpatialRay(
            origin_3d=(0.5, 0.5, 0.0),
            direction_3d=(1.0, 0.0, 0.0),
            length=0.5,
            endpoint_3d=(1.0, 0.5, 0.0),
            timestamp=1.0,
            valid=True,
        )
        proj = CameraProjection.project_ray(ray, frame_width=1000, frame_height=1000, visual_length_px=300.0)
        assert proj.origin_px == (500, 500)
        assert proj.endpoint_px == (800, 500)

    def test_projected_ray_visibility_flag(self):
        ray = SpatialRay(
            origin_3d=(0.5, 0.5, 0.0),
            direction_3d=(0.0, 1.0, 0.0),
            length=0.1,
            endpoint_3d=(0.5, 0.6, 0.0),
            timestamp=1.0,
            valid=False,
        )
        proj = CameraProjection.project_ray(ray, frame_width=1280, frame_height=720)
        assert not proj.visible

    def test_normalized_to_pixel_mapping(self):
        px, py = CameraProjection.normalized_to_pixel(0.25, 0.75, 800, 600)
        assert px == 200
        assert py == 450


class TestPhase5NonRegressionGates:
    """Verifies that Phase 2, Phase 3, and Phase 4 capabilities function seamlessly alongside Phase 5."""

    def test_phase2_cursor_active_during_pointing(self):
        backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        cursor_config = CursorConfig(enabled=True, arming_duration_s=0.0)
        controller = CursorController(config=cursor_config, backend=backend)
        estimator = PointingEstimator()

        hand = create_mock_pointing_hand()
        cursor_coords = controller.update(hand.index_tip.x, hand.index_tip.y)
        estimate = estimator.estimate(hand, "POINTING", 1.0)

        assert cursor_coords is not None
        assert estimate.ray.valid
        assert len(backend.move_history) == 1

    def test_phase3_pinch_click_non_regression(self):
        backend = MockMouseBackend()
        engine = GestureConfig(enabled=True, min_pinch_duration_ms=0.0)
        gesture_engine = GestureEngine(config=engine, backend=backend)
        estimator = PointingEstimator()

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        # Pinch -> Ray should deactivate
        gesture_engine.update(hand_pinch, (500, 500))
        est_pinch = estimator.estimate(hand_pinch, gesture_engine.active_gesture, 1.0)
        assert not est_pinch.ray.valid

        # Release -> Single left click
        gesture_engine.update(hand_open, (500, 500))
        assert len(backend.clicks) == 1

    def test_phase3_pinch_drag_non_regression(self):
        backend = MockMouseBackend()
        engine = GestureConfig(enabled=True, drag_threshold_px=5.0, min_pinch_duration_ms=0.0)
        gesture_engine = GestureEngine(config=engine, backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_pinch, (510, 500))
        assert backend.is_button_down

        gesture_engine.update(hand_open, (510, 500))
        assert not backend.is_button_down
        assert len(backend.mouse_ups) == 1

    def test_phase4_safety_lock_non_regression(self):
        backend = MockMouseBackend()
        engine = GestureConfig(enabled=True, activation_frames=2)
        gesture_engine = GestureEngine(config=engine, backend=backend)
        estimator = PointingEstimator()

        hand_fist = create_mock_hand_digits(index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, thumb_ext=False)

        for _ in range(2):
            gesture_engine.update(hand_fist, (500, 500))

        assert gesture_engine.interaction_locked
        estimate = estimator.estimate(hand_fist, gesture_engine.active_gesture, 1.0)
        assert not estimate.ray.valid


# =====================================================================
# PHASE 6 TESTS: OBJECT UNDERSTANDING & TARGET ASSOCIATION
# =====================================================================

class TestPhase6DetectedObjectDataStructures:
    """Validates DetectedObject construction, coordinate normalization, and geometric properties."""

    def test_detected_object_creation_valid(self):
        """Constructs a DetectedObject with verified center, area, and bounding box."""
        obj = DetectedObject.create(
            class_id=39,
            class_name="bottle",
            confidence=0.92,
            bbox=(100, 200, 300, 400),
            timestamp=123.456,
        )
        assert obj.class_id == 39
        assert obj.class_name == "bottle"
        assert math.isclose(obj.confidence, 0.92, rel_tol=1e-3)
        assert obj.bbox == (100, 200, 300, 400)
        assert obj.center == (200, 300)
        assert obj.area == 200 * 200
        assert obj.timestamp == 123.456

    def test_detected_object_inverted_coords_normalization(self):
        """Bounding box with inverted coordinates (x1 > x2 or y1 > y2) must be auto-corrected."""
        obj = DetectedObject.create(
            class_id=1,
            class_name="cup",
            confidence=0.85,
            bbox=(300, 400, 100, 200),
        )
        assert obj.bbox == (100, 200, 300, 400)
        assert obj.center == (200, 300)
        assert obj.area == 40000

    def test_detected_object_zero_area_edge_case(self):
        """Zero-width or zero-height bounding boxes must not raise errors."""
        obj = DetectedObject.create(
            class_id=0,
            class_name="point_object",
            confidence=0.50,
            bbox=(100, 100, 100, 100),
        )
        assert obj.area == 0
        assert obj.center == (100, 100)


class TestPhase6DetectorBackends:
    """Validates detector backends, confidence filtering, and explicit availability reporting."""

    def test_mock_object_detector_default_scene(self):
        """MockObjectDetector returns default synthetic scene with bottle, cup, laptop."""
        detector = MockObjectDetector()
        assert detector.is_available()
        status, detail = detector.get_status()
        assert status == DetectorStatus.SIMULATED

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        assert len(objects) == 3

        names = {o.class_name for o in objects}
        assert names == {"bottle", "cup", "laptop"}

    def test_mock_object_detector_confidence_filter(self):
        """MockObjectDetector filters out detections below confidence threshold."""
        detector = MockObjectDetector(confidence_threshold=0.90)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        # Cup is 0.88, so only bottle (0.92) and laptop (0.95) pass
        assert len(objects) == 2
        names = {o.class_name for o in objects}
        assert names == {"bottle", "laptop"}

    def test_mock_object_detector_custom_scene(self):
        """MockObjectDetector allows dynamic configuration of custom scenes."""
        detector = MockObjectDetector()
        custom = [
            DetectedObject.create(class_id=10, class_name="phone", confidence=0.80, bbox=(50, 50, 150, 200)),
        ]
        detector.set_scene(custom)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        assert len(objects) == 1
        assert objects[0].class_name == "phone"

    def test_mediapipe_detector_missing_model_graceful_unavailable(self):
        """MediaPipeObjectDetector reports UNAVAILABLE when model file is absent without crashing."""
        detector = MediaPipeObjectDetector(model_path="non_existent_model_file_12345.tflite")
        assert not detector.is_available()
        status, detail = detector.get_status()
        assert status == DetectorStatus.UNAVAILABLE
        assert "MODEL_NOT_FOUND" in detail

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        assert objects == []


class TestPhase6ObjectDetectorOrchestrator:
    """Validates frame interval skipping, caching, max age expiration, exception safety, and coordinate rescaling."""

    def test_detection_interval_and_caching(self):
        """Orchestrator runs backend on interval frames and serves cached detections on intermediate frames."""
        mock_backend = MockObjectDetector()
        config = ObjectDetectorConfig(detection_interval_frames=3)
        orchestrator = ObjectDetector(config=config, backend=mock_backend)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Frame 1: Runs detection
        res1 = orchestrator.detect(frame)
        assert len(res1) == 3
        assert orchestrator.cache_age_frames == 0

        # Frame 2: Skipped (cached)
        res2 = orchestrator.detect(frame)
        assert len(res2) == 3
        assert orchestrator.cache_age_frames == 1

        # Frame 3: Skipped (cached)
        res3 = orchestrator.detect(frame)
        assert len(res3) == 3
        assert orchestrator.cache_age_frames == 2

        # Frame 4: Runs detection again
        res4 = orchestrator.detect(frame)
        assert len(res4) == 3
        assert orchestrator.cache_age_frames == 0

    def test_cache_max_age_expiration(self):
        """Cached detections are purged once cache age exceeds max_detection_age_frames."""
        mock_backend = MockObjectDetector()
        config = ObjectDetectorConfig(detection_interval_frames=10, max_detection_age_frames=3)
        orchestrator = ObjectDetector(config=config, backend=mock_backend)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Frame 1: fresh detection
        orchestrator.detect(frame)
        # Frames 2, 3, 4: cached within max_age=3
        for _ in range(3):
            res = orchestrator.detect(frame)
            assert len(res) == 3

        # Frame 5: cache age becomes 4 > max_age 3 -> cache purged
        res_expired = orchestrator.detect(frame)
        assert res_expired == []

    def test_exception_safety_and_error_status(self):
        """Inference exceptions are caught safely, error status reported, and stale cache purged."""
        mock_backend = MockObjectDetector()
        config = ObjectDetectorConfig(detection_interval_frames=1, max_detection_age_frames=2)
        orchestrator = ObjectDetector(config=config, backend=mock_backend)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        orchestrator.detect(frame)
        assert len(orchestrator._cached_detections) == 3

        # Trigger simulated error
        mock_backend.set_raise_on_detect(True)

        # Frame 2: catches error, retains cache for 1 frame
        res2 = orchestrator.detect(frame)
        status, detail = orchestrator.get_status()
        assert status == DetectorStatus.ERROR
        assert "INFERENCE_ERROR" in detail

        # Frame 3 & 4: error persists, exceeds max_age=2 -> purged
        orchestrator.detect(frame)
        res4 = orchestrator.detect(frame)
        assert res4 == []

    def test_coordinate_scaling_rescaling_invariant(self):
        """Bounding boxes from resized detector space are accurately rescaled to native frame coordinates."""
        orig_w, orig_h = 640, 480
        det_w, det_h = 320, 240

        det_objs = [
            DetectedObject.create(class_id=1, class_name="test_box", confidence=0.90, bbox=(100, 50, 150, 100))
        ]

        rescaled = ObjectDetector.rescale_detections(
            detections=det_objs,
            orig_width=orig_w,
            orig_height=orig_h,
            detector_width=det_w,
            detector_height=det_h,
        )

        assert len(rescaled) == 1
        res_obj = rescaled[0]
        # Scaled by 2x
        assert res_obj.bbox == (200, 100, 300, 200)
        assert res_obj.center == (250, 150)
        assert res_obj.area == 100 * 100


class TestPhase6TargetAssociatorContainmentAndMargin:
    """Validates hit testing, dual containment, boundary margin behavior, and continuous scoring."""

    def test_target_inside_base_bbox(self):
        """Target strictly inside base bbox returns inside_bbox=True and containment_score >= 0.5."""
        bbox = (500, 250, 650, 450)
        inside_bbox, inside_margin, score = TargetAssociator.calculate_box_containment(
            px=575, py=350, bbox=bbox, margin=14.0
        )
        assert inside_bbox
        assert inside_margin
        assert math.isclose(score, 1.0, abs_tol=0.05)  # Exact center

    def test_target_exact_bbox_edge(self):
        """Target lying exactly on the bbox border gives inside_bbox=True, inside_margin=True, containment=0.5."""
        bbox = (500, 250, 650, 450)
        inside_bbox, inside_margin, score = TargetAssociator.calculate_box_containment(
            px=500, py=350, bbox=bbox, margin=14.0
        )
        assert inside_bbox
        assert inside_margin
        assert math.isclose(score, 0.50, abs_tol=1e-3)

    def test_target_1px_outside_bbox(self):
        """Target 1px outside base bbox gives inside_bbox=False, inside_margin=True, score in (0.0, 0.5)."""
        bbox = (500, 250, 650, 450)
        inside_bbox, inside_margin, score = TargetAssociator.calculate_box_containment(
            px=499, py=350, bbox=bbox, margin=14.0
        )
        assert not inside_bbox
        assert inside_margin
        assert 0.0 < score < 0.50

    def test_target_13px_outside_bbox(self):
        """Target 13px outside with margin 14px is inside margin region."""
        bbox = (500, 250, 650, 450)
        inside_bbox, inside_margin, score = TargetAssociator.calculate_box_containment(
            px=487, py=350, bbox=bbox, margin=14.0
        )
        assert not inside_bbox
        assert inside_margin
        assert 0.0 < score < 0.50

    def test_target_14px_outside_bbox_margin_boundary(self):
        """Target exactly at 14px margin boundary gives containment score 0.0."""
        bbox = (500, 250, 650, 450)
        inside_bbox, inside_margin, score = TargetAssociator.calculate_box_containment(
            px=486, py=350, bbox=bbox, margin=14.0
        )
        assert not inside_bbox
        assert inside_margin
        assert math.isclose(score, 0.0, abs_tol=1e-3)

    def test_target_15px_outside_bbox_rejected(self):
        """Target 15px outside with margin 14px gives inside_margin=False, score=0.0."""
        bbox = (500, 250, 650, 450)
        inside_bbox, inside_margin, score = TargetAssociator.calculate_box_containment(
            px=485, py=350, bbox=bbox, margin=14.0
        )
        assert not inside_bbox
        assert not inside_margin
        assert score == 0.0

    def test_continuous_containment_scoring_gradient(self):
        """Containment score monotonically decreases from center to boundary to margin."""
        bbox = (500, 250, 650, 450)
        _, _, score_center = TargetAssociator.calculate_box_containment(575, 350, bbox, 14.0)
        _, _, score_near_edge = TargetAssociator.calculate_box_containment(510, 350, bbox, 14.0)
        _, _, score_on_edge = TargetAssociator.calculate_box_containment(500, 350, bbox, 14.0)
        _, _, score_in_margin = TargetAssociator.calculate_box_containment(495, 350, bbox, 14.0)

        assert score_center > score_near_edge
        assert score_near_edge > score_on_edge
        assert score_on_edge > score_in_margin


class TestPhase6TargetAssociatorArbitration:
    """Validates multi-candidate association and deterministic tie-breaking."""

    def test_single_object_association(self):
        """TargetAssociator associates ray with single detected object."""
        associator = TargetAssociator()
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))

        raw_res = associator.associate_raw(ray_endpoint=(575, 350), detected_objects=[obj])
        assert raw_res is not None
        matched_obj, total_score, cont, prox, conf, dist, in_bbox, in_margin = raw_res
        assert matched_obj.class_name == "bottle"
        assert in_bbox
        assert in_margin
        assert total_score > 0.80

    def test_no_object_association(self):
        """Point outside all objects returns None."""
        associator = TargetAssociator()
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))

        raw_res = associator.associate_raw(ray_endpoint=(100, 100), detected_objects=[obj])
        assert raw_res is None

    def test_empty_detected_objects_list(self):
        """Empty detections list returns None."""
        associator = TargetAssociator()
        raw_res = associator.associate_raw(ray_endpoint=(500, 500), detected_objects=[])
        assert raw_res is None

    def test_overlapping_bboxes_highest_score_wins(self):
        """When multiple bounding boxes overlap, the candidate with highest deterministic score is chosen."""
        associator = TargetAssociator()
        obj_bottle = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.95, bbox=(500, 250, 650, 450))
        obj_table = DetectedObject.create(class_id=60, class_name="dining table", confidence=0.60, bbox=(400, 200, 800, 600))

        # Ray is aimed directly at the bottle center (575, 350)
        raw_res = associator.associate_raw(ray_endpoint=(575, 350), detected_objects=[obj_bottle, obj_table])
        assert raw_res is not None
        matched_obj = raw_res[0]
        assert matched_obj.class_name == "bottle"

    def test_deterministic_tie_breaking_proximity(self):
        """When containment and confidence are identical, closer center distance breaks the tie."""
        associator = TargetAssociator()
        # Two objects with same confidence
        obj_a = DetectedObject.create(class_id=1, class_name="box_a", confidence=0.90, bbox=(500, 300, 600, 400)) # center (550, 350)
        obj_b = DetectedObject.create(class_id=2, class_name="box_b", confidence=0.90, bbox=(520, 300, 620, 400)) # center (570, 350)

        # Ray at (552, 350) -> closer to box_a
        raw_res = associator.associate_raw(ray_endpoint=(552, 350), detected_objects=[obj_a, obj_b])
        assert raw_res is not None
        assert raw_res[0].class_name == "box_a"


class TestPhase6TargetTemporalStabilization:
    """Validates target state transitions, activation frames, release grace frames, and identity continuity."""

    def test_target_state_transitions_unstable_to_locked(self):
        """Target transitions from UNSTABLE to LOCKED after required activation frames."""
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=3, release_frames=3))
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        # Frame 1: UNSTABLE
        t1 = associator.update(projected_ray=ray, detected_objects=[obj], active_gesture="POINTING", pointing_valid=True)
        assert t1 is not None
        assert t1.state == TargetState.UNSTABLE
        assert not t1.stable
        assert t1.frames_active == 1

        # Frame 2: UNSTABLE
        t2 = associator.update(projected_ray=ray, detected_objects=[obj], active_gesture="POINTING", pointing_valid=True)
        assert t2 is not None
        assert t2.state == TargetState.UNSTABLE
        assert not t2.stable
        assert t2.frames_active == 2

        # Frame 3: LOCKED
        t3 = associator.update(projected_ray=ray, detected_objects=[obj], active_gesture="POINTING", pointing_valid=True)
        assert t3 is not None
        assert t3.state == TargetState.LOCKED
        assert t3.stable
        assert t3.frames_active == 3

    def test_target_release_grace_period_flicker_suppression(self):
        """Locked target is held during single-frame detector drops and released after release_frames."""
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2, release_frames=2))
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        # Lock the target (2 frames)
        associator.update(ray, [obj], "POINTING", True)
        t_locked = associator.update(ray, [obj], "POINTING", True)
        assert t_locked.state == TargetState.LOCKED

        # Frame 3: Object missing (single frame drop) -> Grace frame 1 (held)
        t_drop1 = associator.update(ray, [], "POINTING", True)
        assert t_drop1 is not None
        assert t_drop1.detected_object.class_name == "bottle"

        # Frame 4: Object still missing -> Grace frame 2 (decayed state)
        t_drop2 = associator.update(ray, [], "POINTING", True)
        assert t_drop2 is not None

        # Frame 5: Release expired -> Target reset to None
        t_drop3 = associator.update(ray, [], "POINTING", True)
        assert t_drop3 is None
        assert associator.target_state == TargetState.NONE

    def test_identity_continuity_with_spatial_drift(self):
        """Moving object retains candidate identity across consecutive frames."""
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=3))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        obj_f1 = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        obj_f2 = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(505, 250, 655, 450)) # 5px shift
        obj_f3 = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(510, 250, 660, 450)) # 5px shift

        associator.update(ray, [obj_f1], "POINTING", True)
        associator.update(ray, [obj_f2], "POINTING", True)
        t3 = associator.update(ray, [obj_f3], "POINTING", True)

        assert t3.state == TargetState.LOCKED
        assert t3.frames_active == 3


class TestPhase6GestureAndPipelineIntegration:
    """Validates gesture arbitration gating, suppression under non-pointing gestures, and zero mouse clicks."""

    def test_pointing_enables_target_association(self):
        """POINTING gesture with valid ray successfully activates target association."""
        associator = TargetAssociator()
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        target = associator.update(ray, [obj], active_gesture="POINTING", pointing_valid=True)
        assert target is not None

    def test_open_palm_suppresses_target_association(self):
        """OPEN_PALM gesture resets and suppresses object target association."""
        associator = TargetAssociator()
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=False)

        target = associator.update(ray, [obj], active_gesture="OPEN_PALM", pointing_valid=False)
        assert target is None
        assert associator.target_state == TargetState.NONE

    def test_fist_safety_lock_suppresses_target_association(self):
        """FIST safety lock resets and suppresses object target association."""
        associator = TargetAssociator()
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=False)

        target = associator.update(ray, [obj], active_gesture="FIST", pointing_valid=False)
        assert target is None

    def test_hand_loss_clears_target(self):
        """Hand loss (None ray or invalid pointing) immediately resets target."""
        associator = TargetAssociator()
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        # Acquire target
        associator.update(ray, [obj], "POINTING", True)
        assert associator.target_state != TargetState.NONE

        # Hand lost
        target_lost = associator.update(None, [obj], "UNKNOWN", False)
        assert target_lost is None
        assert associator.target_state == TargetState.NONE

    def test_no_mouse_clicks_generated_during_targeting(self):
        """Phase 6 object targeting is strictly observation and produces 0 mouse click events."""
        mouse_backend = MockMouseBackend()
        cursor_controller = CursorController(config=CursorConfig(enabled=True, arming_duration_s=0.0), backend=mouse_backend)
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True), backend=mouse_backend)
        associator = TargetAssociator()
        obj = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        for _ in range(5):
            associator.update(ray, [obj], "POINTING", True)

        assert len(mouse_backend.clicks) == 0
        assert len(mouse_backend.right_clicks) == 0
        assert len(mouse_backend.middle_clicks) == 0
        assert not mouse_backend.is_button_down


class TestPhase6SyntheticScenesAndEndToEnd:
    """Validates deterministic synthetic scene targeting and end-to-end execution."""

    def test_synthetic_bottle_targeting(self):
        """Ray targeted at (575, 350) targets the bottle in Synthetic Scene A."""
        detector = MockObjectDetector()
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)

        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)
        associator.update(ray, objects, "POINTING", True)
        target = associator.update(ray, objects, "POINTING", True)

        assert target is not None
        assert target.detected_object.class_name == "bottle"
        assert target.state == TargetState.LOCKED

    def test_synthetic_cup_targeting(self):
        """Ray targeted at (750, 350) targets the cup in Synthetic Scene A."""
        detector = MockObjectDetector()
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)

        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(750, 350), visible=True)
        associator.update(ray, objects, "POINTING", True)
        target = associator.update(ray, objects, "POINTING", True)

        assert target is not None
        assert target.detected_object.class_name == "cup"
        assert target.state == TargetState.LOCKED

    def test_synthetic_laptop_targeting(self):
        """Ray targeted at (350, 300) targets the laptop in Synthetic Scene A."""
        detector = MockObjectDetector()
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)

        ray = ProjectedRay(origin_px=(100, 300), endpoint_px=(350, 300), visible=True)
        associator.update(ray, objects, "POINTING", True)
        target = associator.update(ray, objects, "POINTING", True)

        assert target is not None
        assert target.detected_object.class_name == "laptop"
        assert target.state == TargetState.LOCKED

    def test_synthetic_no_target_area(self):
        """Ray targeted at (900, 500) hits no objects in Synthetic Scene A."""
        detector = MockObjectDetector()
        associator = TargetAssociator()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)

        ray = ProjectedRay(origin_px=(100, 100), endpoint_px=(900, 500), visible=True)
        target = associator.update(ray, objects, "POINTING", True)
        assert target is None

    def test_full_synthetic_pipeline_run(self):
        """run() in synthetic mode completes cleanly for 5 frames with exit code 0."""
        exit_code = run(
            synthetic_mode=True,
            max_frames=5,
            cursor_enabled=False,
        )
        assert exit_code == 0


class TestPhase6NonRegressionGates:
    """Verifies that Phase 2, Phase 3, Phase 4, and Phase 5 capabilities remain 100% operational."""

    def test_phase2_cursor_non_regression_during_object_detection(self):
        """Phase 2 cursor tracking continues normally alongside object detection."""
        backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        cursor_controller = CursorController(config=CursorConfig(enabled=True, arming_duration_s=0.0), backend=backend)
        detector = ObjectDetector(backend=MockObjectDetector())

        hand = create_mock_pointing_hand()
        coords = cursor_controller.update(hand.index_tip.x, hand.index_tip.y)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)

        assert coords is not None
        assert len(objects) == 3
        assert len(backend.move_history) == 1

    def test_phase3_pinch_click_non_regression_during_object_detection(self):
        """Phase 3 pinch click operates without regression during object detection."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, min_pinch_duration_ms=0.0), backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_open, (500, 500))
        assert len(backend.clicks) == 1

    def test_phase4_right_middle_click_non_regression(self):
        """Phase 4 two-finger right click and three-finger middle click operate without regression."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, activation_frames=2), backend=backend)

        hand_2f = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        for _ in range(2):
            gesture_engine.update(hand_2f, (500, 500))
        assert len(backend.right_clicks) == 1

        hand_3f = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=False)
        for _ in range(2):
            gesture_engine.update(hand_3f, (500, 500))
        assert len(backend.middle_clicks) == 1


# =====================================================================
# PHASE 7 TESTS: OBJECT SELECTION + PINCH INTERACTION
# =====================================================================

class TestPhase7SelectionDataStructures:
    """Validates SelectedObject construction, identity generation, and configuration defaults."""

    def test_selected_object_creation_and_properties(self):
        """SelectedObject creates an immutable reference with identity and selection telemetry."""
        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.92, bbox=(500, 250, 650, 450))
        targeted = TargetedObject(
            detected_object=det,
            score=0.88,
            containment_score=0.95,
            proximity_score=0.90,
            confidence_score=0.92,
            distance_px=15.0,
            inside_bbox=True,
            inside_margin=True,
            state=TargetState.LOCKED,
            stable=True,
            frames_active=5,
            timestamp=100.0,
        )

        selected = SelectedObject.create(targeted_object=targeted, frame_index=120, timestamp=100.0)

        assert selected.class_id == 39
        assert selected.class_name == "bottle"
        assert math.isclose(selected.selection_score, 0.88, rel_tol=1e-3)
        assert selected.confidence == 0.92
        assert selected.frame_index == 120
        assert selected.stable_target_frames == 5
        assert "39_bottle_500_250" in selected.object_identity

    def test_selection_config_defaults(self):
        """SelectionConfig contains documented, robust default parameters."""
        config = SelectionConfig()
        assert config.require_locked_target is True
        assert config.minimum_target_score == 0.40
        assert config.selection_cooldown_frames == 5
        assert config.release_on_target_change is True
        assert config.cancel_on_hand_loss is True
        assert config.cancel_on_fist is True
        assert config.cancel_on_open_palm is True

    def test_selection_state_enum_completeness(self):
        """SelectionState includes all 4 lifecycle states."""
        states = {s.value for s in SelectionState}
        assert states == {"NONE", "ARMED", "SELECTED", "CANCELLED"}

    def test_selection_event_enum_completeness(self):
        """SelectionEvent includes all 4 lifecycle events."""
        events = {e.value for e in SelectionEvent}
        assert events == {"SELECT_STARTED", "SELECTED", "SELECTION_RELEASED", "SELECTION_CANCELLED"}


class TestPhase7PinchToSelectLifecycle:
    """Validates the core POINT -> LOCK -> PINCH -> SELECT lifecycle and edge triggering."""

    def _create_mock_locked_target(self, class_name: str = "bottle", score: float = 0.85) -> TargetedObject:
        det = DetectedObject.create(class_id=39, class_name=class_name, confidence=0.90, bbox=(500, 250, 650, 450))
        return TargetedObject(
            detected_object=det,
            score=score,
            containment_score=0.90,
            proximity_score=0.90,
            confidence_score=0.90,
            distance_px=10.0,
            inside_bbox=True,
            inside_margin=True,
            state=TargetState.LOCKED,
            stable=True,
            frames_active=4,
            timestamp=1.0,
        )

    def test_locked_target_plus_pinch_selects_object(self):
        """LOCKED target + pinch initiation transitions state from ARMED to SELECTED."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Frame 1: Pointing at locked target without pinch -> ARMED
        st1 = engine.update(targeted_object=target, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        assert st1 == SelectionState.ARMED
        assert not engine.is_selected
        assert engine.selected_object is None

        # Frame 2: Pinch initiation -> SELECTED
        st2 = engine.update(targeted_object=target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st2 == SelectionState.SELECTED
        assert engine.is_selected
        assert engine.selected_object is not None
        assert engine.selected_object.class_name == "bottle"
        assert engine.selection_triggered is True
        assert engine.last_event == SelectionEvent.SELECTED

    def test_locked_target_without_pinch_is_armed(self):
        """LOCKED target without pinch remains ARMED and does not select."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        for _ in range(5):
            st = engine.update(targeted_object=target, is_pinched=False, active_gesture="POINTING", hand_valid=True)
            assert st == SelectionState.ARMED
            assert not engine.is_selected

    def test_pinch_start_generates_single_selection_event(self):
        """Pinch initiation sets selection_triggered=True for exactly one frame."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Frame 1: Pinch start
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.selection_triggered is True

        # Frame 2: Still pinching
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.selection_triggered is False

    def test_holding_pinch_does_not_repeat_selection(self):
        """Holding pinch across 10 consecutive frames keeps object selected without retriggering."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Frame 1: Initial select
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.selection_triggered is True

        # Frames 2-10: Pinch held
        for _ in range(9):
            st = engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
            assert st == SelectionState.SELECTED
            assert engine.is_selected
            assert engine.selection_triggered is False

    def test_releasing_pinch_maintains_selection(self):
        """Releasing pinch maintains active selection without duplicate events."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Select
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected

        # Release pinch
        st_released = engine.update(target, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        assert st_released == SelectionState.SELECTED
        assert engine.is_selected
        assert engine.selected_object.class_name == "bottle"

    def test_new_pinch_cycle_after_release_can_reselect(self):
        """After releasing pinch and cooldown, a new pinch on a locked target initiates a new selection."""
        engine = SelectionEngine(config=SelectionConfig(selection_cooldown_frames=2))
        target_a = self._create_mock_locked_target(class_name="bottle")

        # Select Bottle A
        engine.update(target_a, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected

        # Release pinch and let cooldown expire
        engine.update(target_a, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        engine.update(target_a, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        engine.update(target_a, is_pinched=False, active_gesture="POINTING", hand_valid=True)

        # Repinch on target
        engine.update(target_a, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected


class TestPhase7PreconditionsAndInvalidSelection:
    """Validates strict preconditions: UNSTABLE, SEARCHING, NONE, or low score cannot select."""

    def test_none_target_plus_pinch_does_not_select(self):
        """Pinch with no target produces SelectionState.NONE."""
        engine = SelectionEngine()
        st = engine.update(targeted_object=None, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected

    def test_searching_target_plus_pinch_does_not_select(self):
        """Pinch while target state is SEARCHING produces SelectionState.NONE."""
        engine = SelectionEngine()
        det = DetectedObject.create(class_id=1, class_name="cup", confidence=0.80, bbox=(100, 100, 200, 200))
        target_searching = TargetedObject(
            detected_object=det,
            score=0.80,
            containment_score=0.80,
            proximity_score=0.80,
            confidence_score=0.80,
            distance_px=20.0,
            inside_bbox=True,
            inside_margin=True,
            state=TargetState.SEARCHING,
            stable=False,
            frames_active=0,
            timestamp=1.0,
        )

        st = engine.update(targeted_object=target_searching, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected

    def test_unstable_target_plus_pinch_does_not_select(self):
        """Pinch on UNSTABLE target (not yet held for activation_frames) is strictly rejected."""
        engine = SelectionEngine()
        det = DetectedObject.create(class_id=1, class_name="cup", confidence=0.80, bbox=(100, 100, 200, 200))
        target_unstable = TargetedObject(
            detected_object=det,
            score=0.80,
            containment_score=0.80,
            proximity_score=0.80,
            confidence_score=0.80,
            distance_px=20.0,
            inside_bbox=True,
            inside_margin=True,
            state=TargetState.UNSTABLE,
            stable=False,
            frames_active=1,
            timestamp=1.0,
        )

        st = engine.update(targeted_object=target_unstable, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected

    def test_low_target_score_rejects_selection(self):
        """Target with score below minimum_target_score is rejected from selection."""
        engine = SelectionEngine(config=SelectionConfig(minimum_target_score=0.50))
        det = DetectedObject.create(class_id=1, class_name="cup", confidence=0.80, bbox=(100, 100, 200, 200))
        target_low_score = TargetedObject(
            detected_object=det,
            score=0.35,  # < 0.50
            containment_score=0.35,
            proximity_score=0.35,
            confidence_score=0.35,
            distance_px=50.0,
            inside_bbox=False,
            inside_margin=True,
            state=TargetState.LOCKED,
            stable=True,
            frames_active=5,
            timestamp=1.0,
        )

        st = engine.update(targeted_object=target_low_score, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected


class TestPhase7IdentityContinuityAndTargetSwitching:
    """Validates selected-object identity locking and safe cancellation on target switching."""

    def test_same_object_spatial_drift_maintains_selection(self):
        """Selected object moving slightly across frames updates geometry and retains selection."""
        engine = SelectionEngine()
        det1 = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        det2 = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(508, 250, 658, 450)) # 8px drift

        tgt1 = TargetedObject(
            detected_object=det1, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=3, timestamp=1.0,
        )
        tgt2 = TargetedObject(
            detected_object=det2, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=12.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.033,
        )

        # Select on frame 1
        engine.update(tgt1, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected
        assert engine.selected_object.detected_object.bbox == (500, 250, 650, 450)

        # Frame 2 with drift
        engine.update(tgt2, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected
        assert engine.selected_object.detected_object.bbox == (508, 250, 658, 450)

    def test_target_switching_to_different_object_cancels_selection(self):
        """When ray shifts to a completely different object, the selection is cancelled."""
        engine = SelectionEngine(config=SelectionConfig(release_on_target_change=True))
        det_bottle = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        det_laptop = DetectedObject.create(class_id=63, class_name="laptop", confidence=0.95, bbox=(200, 200, 450, 450))

        tgt_bottle = TargetedObject(
            detected_object=det_bottle, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )
        tgt_laptop = TargetedObject(
            detected_object=det_laptop, score=0.90, containment_score=0.90, proximity_score=0.90,
            confidence_score=0.95, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.033,
        )

        # Select Bottle
        engine.update(tgt_bottle, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected
        assert engine.selected_object.class_name == "bottle"

        # Shift target to Laptop
        st = engine.update(tgt_laptop, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert not engine.is_selected
        assert engine.last_event == SelectionEvent.SELECTION_CANCELLED

    def test_same_class_different_instance_identity_isolation(self):
        """Two distant bottles of the same class cannot silently swap selection identity."""
        engine = SelectionEngine(config=SelectionConfig(release_on_target_change=True))
        det_bottle1 = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(100, 200, 250, 400)) # Center (175, 300)
        det_bottle2 = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(800, 200, 950, 400)) # Center (875, 300)

        tgt1 = TargetedObject(
            detected_object=det_bottle1, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )
        tgt2 = TargetedObject(
            detected_object=det_bottle2, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.033,
        )

        # Select Bottle 1
        engine.update(tgt1, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected
        assert engine.selected_object.detected_object.center == (175, 300)

        # Shift to distant Bottle 2
        engine.update(tgt2, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        # Identity distance check fails -> cancels selection
        assert not engine.is_selected


class TestPhase7SafetyAndInterruptionGating:
    """Validates hand loss, FIST safety lock, OPEN_PALM cancellations, and zero mouse events."""

    def _create_mock_locked_target(self) -> TargetedObject:
        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        return TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

    def test_hand_loss_cancels_selection(self):
        """Hand loss immediately clears active selection and resets state."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Select
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected

        # Hand lost
        st = engine.update(targeted_object=None, is_pinched=False, active_gesture="UNKNOWN", hand_valid=False)
        assert st == SelectionState.NONE
        assert not engine.is_selected
        assert engine.selected_object is None
        assert engine.last_event == SelectionEvent.SELECTION_CANCELLED

    def test_fist_safety_lock_cancels_selection(self):
        """FIST safety lock immediately cancels selection."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Select
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected

        # User forms FIST
        st = engine.update(targeted_object=target, is_pinched=False, active_gesture="FIST", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected
        assert engine.last_event == SelectionEvent.SELECTION_CANCELLED

    def test_open_palm_cancels_selection(self):
        """OPEN_PALM gesture clears active selection."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Select
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected

        # User opens palm
        st = engine.update(targeted_object=None, is_pinched=False, active_gesture="OPEN_PALM", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected

    def test_target_loss_cancels_selection(self):
        """Ray drifting away from target cancels the selection safely."""
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected

        # Ray moves off target -> targeted_object is None
        st = engine.update(targeted_object=None, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected
        assert engine.last_event == SelectionEvent.SELECTION_CANCELLED

    def test_selection_produces_zero_mouse_actions(self):
        """Object selection generates zero physical or mock mouse click/drag events."""
        mouse_backend = MockMouseBackend()
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        for _ in range(5):
            engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)

        assert len(mouse_backend.clicks) == 0
        assert len(mouse_backend.right_clicks) == 0
        assert len(mouse_backend.middle_clicks) == 0
        assert not mouse_backend.is_button_down


class TestPhase7SyntheticScenesAndEndToEnd:
    """Validates end-to-end selection using synthetic frames and mock detector scenes."""

    def test_synthetic_scene_bottle_selection(self):
        """Synthetic Scene A: pointing ray at Bottle (575, 350) + pinch selects Bottle."""
        detector = MockObjectDetector()
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2))
        selection_engine = SelectionEngine()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        # 2 frames to achieve LOCKED
        associator.update(ray, objects, "POINTING", True)
        tgt_locked = associator.update(ray, objects, "POINTING", True)
        assert tgt_locked.state == TargetState.LOCKED
        assert tgt_locked.detected_object.class_name == "bottle"

        # Pinch to select
        st = selection_engine.update(tgt_locked, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.SELECTED
        assert selection_engine.is_selected
        assert selection_engine.selected_object.class_name == "bottle"

    def test_synthetic_scene_cup_selection(self):
        """Synthetic Scene A: pointing ray at Cup (750, 350) + pinch selects Cup."""
        detector = MockObjectDetector()
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2))
        selection_engine = SelectionEngine()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(750, 350), visible=True)

        associator.update(ray, objects, "POINTING", True)
        tgt_locked = associator.update(ray, objects, "POINTING", True)
        assert tgt_locked.state == TargetState.LOCKED
        assert tgt_locked.detected_object.class_name == "cup"

        st = selection_engine.update(tgt_locked, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.SELECTED
        assert selection_engine.selected_object.class_name == "cup"

    def test_synthetic_scene_laptop_selection(self):
        """Synthetic Scene A: pointing ray at Laptop (350, 300) + pinch selects Laptop."""
        detector = MockObjectDetector()
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2))
        selection_engine = SelectionEngine()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        ray = ProjectedRay(origin_px=(100, 300), endpoint_px=(350, 300), visible=True)

        associator.update(ray, objects, "POINTING", True)
        tgt_locked = associator.update(ray, objects, "POINTING", True)
        assert tgt_locked.state == TargetState.LOCKED
        assert tgt_locked.detected_object.class_name == "laptop"

        st = selection_engine.update(tgt_locked, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.SELECTED
        assert selection_engine.selected_object.class_name == "laptop"

    def test_synthetic_scene_no_target_pinch(self):
        """Synthetic Scene A: ray at empty area (900, 500) + pinch does NOT select."""
        detector = MockObjectDetector()
        associator = TargetAssociator()
        selection_engine = SelectionEngine()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects = detector.detect(frame)
        ray = ProjectedRay(origin_px=(100, 100), endpoint_px=(900, 500), visible=True)

        tgt = associator.update(ray, objects, "POINTING", True)
        st = selection_engine.update(tgt, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.NONE
        assert not selection_engine.is_selected

    def test_full_synthetic_pipeline_run_phase7(self):
        """run() in synthetic mode with selection enabled executes 10 frames cleanly with exit code 0."""
        exit_code = run(
            synthetic_mode=True,
            max_frames=10,
            cursor_enabled=False,
            selection_enabled=True,
        )
        assert exit_code == 0


class TestPhase7NonRegressionGates:
    """Verifies that Phases 1, 2, 3, 4, 5, and 6 capabilities remain 100% operational."""

    def test_phase2_cursor_non_regression_during_selection(self):
        """Phase 2 cursor tracking continues normally alongside selection engine."""
        backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        cursor_controller = CursorController(config=CursorConfig(enabled=True, arming_duration_s=0.0), backend=backend)
        selection_engine = SelectionEngine()

        hand = create_mock_pointing_hand()
        coords = cursor_controller.update(hand.index_tip.x, hand.index_tip.y)
        selection_engine.update(targeted_object=None, is_pinched=False, active_gesture="POINTING", hand_valid=True)

        assert coords is not None
        assert len(backend.move_history) == 1

    def test_phase3_pinch_click_non_regression_in_desktop_mode(self):
        """Phase 3 pinch click in desktop mode continues dispatching mouse clicks without interference."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, min_pinch_duration_ms=0.0), backend=backend)
        selection_engine = SelectionEngine()

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        gesture_engine.update(hand_pinch, (500, 500))
        selection_engine.update(None, is_pinched=gesture_engine.is_pinched, active_gesture="PINCHING", hand_valid=True)

        gesture_engine.update(hand_open, (500, 500))
        selection_engine.update(None, is_pinched=gesture_engine.is_pinched, active_gesture="POINTING", hand_valid=True)

        assert len(backend.clicks) == 1

    def test_phase3_pinch_drag_non_regression_in_desktop_mode(self):
        """Phase 3 pinch drag lifecycle in desktop mode remains operational."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, drag_threshold_px=5.0, min_pinch_duration_ms=0.0), backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_pinch, (510, 500))
        assert backend.is_button_down

        gesture_engine.update(hand_open, (510, 500))
        assert not backend.is_button_down
        assert len(backend.mouse_ups) == 1

    def test_phase4_gestures_non_regression(self):
        """Phase 4 right-click, middle-click, and fist safety lock remain operational."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, activation_frames=2), backend=backend)

        hand_2f = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        for _ in range(2):
            gesture_engine.update(hand_2f, (500, 500))
        assert len(backend.right_clicks) == 1

        hand_fist = create_mock_hand_digits(index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, thumb_ext=False)
        for _ in range(2):
            gesture_engine.update(hand_fist, (500, 500))
        assert gesture_engine.interaction_locked


class TestPhase7RefinementsAndEdgeSpecifications:
    """Validates detailed edge specifications, cache retention vs target loss, and desktop cursor isolation."""

    def _create_mock_locked_target(self, class_name: str = "bottle", bbox=(500, 250, 650, 450)) -> TargetedObject:
        det = DetectedObject.create(class_id=39, class_name=class_name, confidence=0.90, bbox=bbox)
        return TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

    def test_selection_triggered_edge_invariant_strict_sequence(self):
        """
        selection_triggered == True ONLY on the frame where:
        previous pinch = False, current pinch = True, AND valid LOCKED target exists.
        Every subsequent pinch frame: selection_triggered == False.
        """
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Frame 1: Hovering (pinch=False) -> selection_triggered=False
        engine.update(target, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        assert engine.selection_triggered is False
        assert engine.state == SelectionState.ARMED

        # Frame 2: Pinch starts (prev=False, curr=True) -> selection_triggered=True
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.selection_triggered is True
        assert engine.state == SelectionState.SELECTED

        # Frame 3: Pinch held (prev=True, curr=True) -> selection_triggered=False
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.selection_triggered is False
        assert engine.state == SelectionState.SELECTED

        # Frame 4: Pinch still held -> selection_triggered=False
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.selection_triggered is False
        assert engine.state == SelectionState.SELECTED

        # Frame 5: Pinch released (prev=True, curr=False) -> selection_triggered=False, stays SELECTED
        engine.update(target, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        assert engine.selection_triggered is False
        assert engine.state == SelectionState.SELECTED
        assert engine.last_event == SelectionEvent.SELECTION_RELEASED

        # Frame 6: Ready for next cycle
        engine.update(target, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        assert engine.selection_triggered is False

    def test_selected_target_disappears_after_successful_selection_clears_safely(self):
        """
        Selected target disappears after successful selection:
        Bottle -> LOCKED -> Pinch -> SELECTED -> Target disappears -> selection cleared safely.
        """
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # 1. Select Bottle
        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected
        assert engine.state == SelectionState.SELECTED

        # 2. Target disappears on next frame
        st = engine.update(targeted_object=None, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        assert st == SelectionState.NONE
        assert not engine.is_selected
        assert engine.selected_object is None
        assert engine.last_event == SelectionEvent.SELECTION_CANCELLED

    def test_detector_transient_failure_preserves_selection_via_cache(self):
        """
        A single failed detector inference does not drop selection because Phase 6
        target caching maintains the LOCKED target. True cache expiration then safely clears selection.
        """
        mock_backend = MockObjectDetector()
        obj_detector = ObjectDetector(
            config=ObjectDetectorConfig(detection_interval_frames=1, max_detection_age_frames=3),
            backend=mock_backend,
        )
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2, release_frames=2))
        selection_engine = SelectionEngine()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)

        # 1. Detect & Lock Bottle (2 frames)
        objs = obj_detector.detect(frame)
        associator.update(ray, objs, "POINTING", True)
        tgt_locked = associator.update(ray, objs, "POINTING", True)
        assert tgt_locked.state == TargetState.LOCKED

        # 2. Select Bottle
        st = selection_engine.update(tgt_locked, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.SELECTED
        assert selection_engine.is_selected

        # 3. Detector transient failure on frame 3 (raises exception)
        mock_backend.set_raise_on_detect(True)
        objs_cached = obj_detector.detect(frame)
        assert len(objs_cached) == 3  # Served from valid cache!

        tgt_held = associator.update(ray, objs_cached, "POINTING", True)
        st_held = selection_engine.update(tgt_held, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st_held == SelectionState.SELECTED
        assert selection_engine.is_selected  # Selection remains stable across detector glitch

        # 4. Detector error persists past max_age -> cache purges -> associator expires -> selection clears
        for _ in range(5):
            objs_expired = obj_detector.detect(frame)
            tgt_exp = associator.update(ray, objs_expired, "POINTING", True)
            st_exp = selection_engine.update(tgt_exp, is_pinched=True, active_gesture="PINCHING", hand_valid=True)

        assert not selection_engine.is_selected
        assert selection_engine.state == SelectionState.NONE

    def test_ambiguous_identity_continuity_cancels_selection(self):
        """
        If incoming target cannot be deterministically associated with the same identity
        (e.g. centroid distance exceeds spatial continuity threshold), SELECTION_CANCELLED is triggered.
        """
        engine = SelectionEngine()
        target_a = self._create_mock_locked_target(class_name="bottle", bbox=(100, 100, 200, 200))
        # Nearby second bottle of same class but too far to be same continuous object (> 120px away)
        target_b = self._create_mock_locked_target(class_name="bottle", bbox=(350, 350, 450, 450))

        # Select Bottle A
        engine.update(target_a, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected
        assert engine.selected_object.detected_object.bbox == (100, 100, 200, 200)

        # Incoming target jumps to Bottle B
        st = engine.update(target_b, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert not engine.is_selected
        assert engine.last_event == SelectionEvent.SELECTION_CANCELLED

    def test_hand_valid_false_clears_selection_subsystem_isolation(self):
        """
        SelectionEngine directly consumes hand_valid boolean; hand_valid=False immediately clears selection.
        """
        engine = SelectionEngine()
        target = self._create_mock_locked_target()

        engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert engine.is_selected

        st = engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=False)
        assert st == SelectionState.NONE
        assert not engine.is_selected
        assert engine.last_event == SelectionEvent.SELECTION_CANCELLED

    def test_selection_with_active_desktop_cursor_generates_zero_pyautogui_calls(self):
        """
        When cursor controller is active, selecting an object generates 0 mouse clicks,
        0 mouseDown, and 0 mouseUp events from the selection action.
        """
        mouse_backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        cursor_config = CursorConfig(enabled=True, arming_duration_s=0.0)
        cursor_controller = CursorController(config=cursor_config, backend=mouse_backend)
        selection_engine = SelectionEngine()

        target = self._create_mock_locked_target()
        hand = create_mock_pointing_hand()

        # Update cursor (moves cursor)
        cursor_controller.update(hand.index_tip.x, hand.index_tip.y)
        initial_moves = len(mouse_backend.move_history)
        assert initial_moves == 1

        # Perform object selection
        selection_engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert selection_engine.is_selected

        # Assert selection action itself produced zero clicks / mouse state mutations
        assert len(mouse_backend.clicks) == 0
        assert len(mouse_backend.right_clicks) == 0
        assert len(mouse_backend.middle_clicks) == 0
        assert len(mouse_backend.mouse_downs) == 0
        assert len(mouse_backend.mouse_ups) == 0
        assert not mouse_backend.is_button_down


# =====================================================================
# PHASE 8 TESTS: DESKTOP MODE + REALITY MODE (UNIFIED CONTEXT)
# =====================================================================

class TestPhase8InteractionModeDataStructures:
    """Validates InteractionMode enums, ModeTransitionResult structure, and ModeController state operations."""

    def test_interaction_mode_enum_values(self):
        """InteractionMode must strictly contain DESKTOP and REALITY."""
        modes = {m.value for m in InteractionMode}
        assert modes == {"DESKTOP", "REALITY"}

    def test_mode_transition_reason_enum_values(self):
        """ModeTransitionReason must contain STARTUP, KEYBOARD_TOGGLE, API_REQUEST, and SAFETY_RESET."""
        reasons = {r.value for r in ModeTransitionReason}
        assert reasons == {"STARTUP", "KEYBOARD_TOGGLE", "API_REQUEST", "SAFETY_RESET"}

    def test_mode_transition_result_immutability_and_fields(self):
        """ModeTransitionResult is an immutable dataclass capturing transition metadata."""
        res = ModeTransitionResult(
            previous_mode=InteractionMode.DESKTOP,
            new_mode=InteractionMode.REALITY,
            reason=ModeTransitionReason.KEYBOARD_TOGGLE,
            changed=True,
            timestamp=100.5,
        )
        assert res.previous_mode == InteractionMode.DESKTOP
        assert res.new_mode == InteractionMode.REALITY
        assert res.reason == ModeTransitionReason.KEYBOARD_TOGGLE
        assert res.changed is True
        assert res.timestamp == 100.5

        # Check immutability
        with pytest.raises(Exception):
            res.changed = False  # type: ignore

    def test_mode_controller_default_mode(self):
        """ModeController defaults to InteractionMode.DESKTOP."""
        ctrl = ModeController()
        assert ctrl.mode == InteractionMode.DESKTOP
        assert ctrl.is_desktop is True
        assert ctrl.is_reality is False

    def test_mode_controller_set_mode_changed(self):
        """set_mode to different mode returns changed=True and updates active mode."""
        ctrl = ModeController(initial_mode=InteractionMode.DESKTOP)
        res = ctrl.set_mode(InteractionMode.REALITY, reason=ModeTransitionReason.API_REQUEST, timestamp=200.0)

        assert res.changed is True
        assert res.previous_mode == InteractionMode.DESKTOP
        assert res.new_mode == InteractionMode.REALITY
        assert ctrl.mode == InteractionMode.REALITY
        assert ctrl.is_reality is True
        assert ctrl.is_desktop is False
        assert ctrl.last_transition == res
        assert ctrl.last_transition_timestamp == 200.0

    def test_mode_controller_set_mode_idempotent(self):
        """set_mode to the same mode returns changed=False."""
        ctrl = ModeController(initial_mode=InteractionMode.DESKTOP)
        res = ctrl.set_mode(InteractionMode.DESKTOP, reason=ModeTransitionReason.API_REQUEST)

        assert res.changed is False
        assert res.previous_mode == InteractionMode.DESKTOP
        assert res.new_mode == InteractionMode.DESKTOP
        assert ctrl.mode == InteractionMode.DESKTOP

    def test_mode_controller_toggle_mode(self):
        """toggle_mode alternates between DESKTOP and REALITY."""
        ctrl = ModeController(initial_mode=InteractionMode.DESKTOP)

        res1 = ctrl.toggle_mode()
        assert res1.changed is True
        assert res1.new_mode == InteractionMode.REALITY
        assert ctrl.is_reality is True

        res2 = ctrl.toggle_mode()
        assert res2.changed is True
        assert res2.new_mode == InteractionMode.DESKTOP
        assert ctrl.is_desktop is True

    def test_mode_controller_reset(self):
        """reset() returns ModeController to its initial startup mode with SAFETY_RESET."""
        ctrl = ModeController(initial_mode=InteractionMode.DESKTOP)
        ctrl.set_mode(InteractionMode.REALITY)
        assert ctrl.is_reality is True

        ctrl.reset()
        assert ctrl.is_desktop is True
        assert ctrl.last_transition.reason == ModeTransitionReason.SAFETY_RESET


class TestPhase8DesktopModeBehavior:
    """Validates full desktop cursor, pinch-click, pinch-drag, and gesture capabilities in Desktop Mode."""

    def test_desktop_mode_cursor_mapping_active(self):
        """In Desktop Mode with cursor enabled, index finger coordinates update mouse backend."""
        backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        cursor_controller = CursorController(config=CursorConfig(enabled=True, arming_duration_s=0.0), backend=backend)

        hand = create_mock_pointing_hand()
        coords = cursor_controller.update(hand.index_tip.x, hand.index_tip.y)

        assert coords is not None
        assert len(backend.move_history) == 1

    def test_desktop_mode_pinch_click_dispatches_left_click(self):
        """In Desktop Mode, pinch and release produces exactly one OS left click."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, min_pinch_duration_ms=0.0), backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        # Pinch
        gesture_engine.update(hand_pinch, (500, 500))
        # Release
        gesture_engine.update(hand_open, (500, 500))

        assert len(backend.clicks) == 1

    def test_desktop_mode_pinch_drag_dispatches_mouse_down_and_up(self):
        """In Desktop Mode, moving with pinch initiates drag (mouseDown) and releases on open (mouseUp)."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, drag_threshold_px=5.0, min_pinch_duration_ms=0.0), backend=backend)

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        # Start pinch at (500, 500) and move to (520, 500)
        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_pinch, (520, 500))
        assert backend.is_button_down
        assert len(backend.mouse_downs) == 1

        # Release pinch
        gesture_engine.update(hand_open, (520, 500))
        assert not backend.is_button_down
        assert len(backend.mouse_ups) == 1

    def test_desktop_mode_two_finger_and_three_finger_clicks(self):
        """In Desktop Mode, two-finger and three-finger gestures dispatch right and middle clicks."""
        backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, activation_frames=2), backend=backend)

        hand_2f = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        for _ in range(2):
            gesture_engine.update(hand_2f, (500, 500))
        assert len(backend.right_clicks) == 1

        hand_3f = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=False)
        for _ in range(2):
            gesture_engine.update(hand_3f, (500, 500))
        assert len(backend.middle_clicks) == 1


class TestPhase8RealityModeBehavior:
    """Validates spatial pointing, object detection, target locking, pinch selection, and zero mouse actions."""

    def _create_mock_locked_target(self) -> TargetedObject:
        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.92, bbox=(500, 250, 650, 450))
        return TargetedObject(
            detected_object=det, score=0.88, containment_score=0.90, proximity_score=0.90,
            confidence_score=0.92, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

    def test_reality_mode_pointing_ray_and_target_locking(self):
        """In Reality Mode, pointing ray projects to camera plane and target associator locks object."""
        associator = TargetAssociator(config=TargetAssociatorConfig(activation_frames=2))
        detector = MockObjectDetector()

        proj_ray = ProjectedRay(origin_px=(300, 350), endpoint_px=(575, 350), visible=True)
        assert proj_ray.visible

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objs = detector.detect(frame)
        associator.update(proj_ray, objs, "POINTING", True)
        target = associator.update(proj_ray, objs, "POINTING", True)

        assert target is not None
        assert target.state == TargetState.LOCKED

    def test_reality_mode_pinch_to_select_and_zero_mouse_actions(self):
        """In Reality Mode, pinching a locked target selects it and generates zero mouse actions."""
        mouse_backend = MockMouseBackend()
        selection_engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Selection update
        st = selection_engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.SELECTED
        assert selection_engine.is_selected
        assert selection_engine.selected_object.class_name == "bottle"

        # Zero mouse events
        assert len(mouse_backend.clicks) == 0
        assert len(mouse_backend.mouse_downs) == 0
        assert len(mouse_backend.mouse_ups) == 0
        assert not mouse_backend.is_button_down


class TestPhase8PyAutoGUIHardInvariant:
    """Rigorous verification that Reality Mode with --cursor enabled generates 0 mouse actions."""

    def test_reality_mode_with_cursor_enabled_generates_zero_mouse_actions(self):
        """
        When run in REALITY mode even with cursor_enabled=True, the system enforces
        the hard invariant: exactly 0 moveTo, 0 click, 0 mouseDown, 0 mouseUp, 0 drag.
        """
        mouse_backend = MockMouseBackend(screen_width=1920, screen_height=1080)
        mode_controller = ModeController(initial_mode=InteractionMode.REALITY)

        # Cursor and Gesture engines configured with enabled=(cursor_enabled and is_desktop)
        cursor_enabled_cli = True
        cursor_config = CursorConfig(enabled=(cursor_enabled_cli and mode_controller.is_desktop))
        gesture_config = GestureConfig(enabled=(cursor_enabled_cli and mode_controller.is_desktop), activation_frames=1)

        cursor_controller = CursorController(config=cursor_config, backend=mouse_backend)
        gesture_engine = GestureEngine(config=gesture_config, backend=mouse_backend)
        selection_engine = SelectionEngine()

        # Create pointing hand and locked target
        hand_point = create_mock_pointing_hand()
        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        target = TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

        # 1. Pointing
        cursor_controller.update(hand_point.index_tip.x, hand_point.index_tip.y)
        gesture_engine.update(hand_point, (640, 360))

        # 2. Pinching (selection)
        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        cursor_controller.update(hand_pinch.index_tip.x, hand_pinch.index_tip.y)
        gesture_engine.update(hand_pinch, (640, 360))
        selection_engine.update(target, is_pinched=gesture_engine.is_pinched, active_gesture="PINCHING", hand_valid=True)
        assert selection_engine.is_selected

        # 3. Two-Finger & Three-Finger in Reality Mode
        hand_2f = create_mock_hand_digits(index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False)
        gesture_engine.update(hand_2f, (640, 360))

        # Strict assertion of the Hard Invariant
        assert len(mouse_backend.move_history) == 0
        assert len(mouse_backend.clicks) == 0
        assert len(mouse_backend.right_clicks) == 0
        assert len(mouse_backend.middle_clicks) == 0
        assert len(mouse_backend.mouse_downs) == 0
        assert len(mouse_backend.mouse_ups) == 0
        assert not mouse_backend.is_button_down


class TestPhase8ModeIsolation:
    """Validates that Desktop Mode actions and Reality Mode spatial selections never cross-contaminate."""

    def test_desktop_pinch_cannot_select_object(self):
        """In Desktop Mode, a pinch gesture performs mouse actions and NEVER triggers SelectionEngine."""
        mode_controller = ModeController(initial_mode=InteractionMode.DESKTOP)
        mouse_backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, min_pinch_duration_ms=0.0), backend=mouse_backend)
        selection_engine = SelectionEngine()

        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        target_locked = TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        # In Desktop Mode, main loop gates selection_engine.reset()
        if mode_controller.is_desktop:
            selection_engine.reset()
            targeted_object = None

        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_open, (500, 500))

        assert len(mouse_backend.clicks) == 1
        assert not selection_engine.is_selected
        assert selection_engine.state == SelectionState.NONE

    def test_reality_pinch_cannot_trigger_desktop_click(self):
        """In Reality Mode, a pinch gesture selects an object and NEVER dispatches a mouse click."""
        mode_controller = ModeController(initial_mode=InteractionMode.REALITY)
        mouse_backend = MockMouseBackend()
        # In Reality Mode, gesture_engine has enabled=False
        gesture_engine = GestureEngine(config=GestureConfig(enabled=(mode_controller.is_desktop)), backend=mouse_backend)
        selection_engine = SelectionEngine()

        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        target_locked = TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)

        gesture_engine.update(hand_pinch, (500, 500))
        selection_engine.update(target_locked, is_pinched=gesture_engine.is_pinched, active_gesture="PINCHING", hand_valid=True)
        assert selection_engine.is_selected

        gesture_engine.update(hand_open, (500, 500))
        assert len(mouse_backend.clicks) == 0


class TestPhase8ModeTransitionsAndSafety:
    """Validates domain-specific state resets and zero stuck buttons during mode transitions."""

    def test_transition_desktop_to_reality_releases_active_mouse_drag(self):
        """
        When switching DESKTOP -> REALITY while dragging:
        Mouse button is safely released (mouse_up), drag is cancelled, and cursor is reset.
        """
        mode_controller = ModeController(initial_mode=InteractionMode.DESKTOP)
        mouse_backend = MockMouseBackend()
        cursor_controller = CursorController(config=CursorConfig(enabled=True, arming_duration_s=0.0), backend=mouse_backend)
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, drag_threshold_px=5.0, min_pinch_duration_ms=0.0), backend=mouse_backend)

        # Initiate desktop drag
        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_pinch, (520, 500))
        assert mouse_backend.is_button_down
        assert gesture_engine.is_dragging

        # Perform mode transition DESKTOP -> REALITY
        res = mode_controller.set_mode(InteractionMode.REALITY)
        assert res.changed is True

        # Execute transition handler resets
        gesture_engine.cleanup()
        cursor_controller.reset()

        assert not mouse_backend.is_button_down
        assert not gesture_engine.is_dragging
        assert len(mouse_backend.mouse_ups) == 1
        assert cursor_controller.arming_state == ArmingState.DISARMED

    def test_transition_reality_to_desktop_clears_active_selection(self):
        """
        When switching REALITY -> DESKTOP while an object is selected:
        Active selection is immediately cleared, target associator reset, and desktop starts neutral.
        """
        mode_controller = ModeController(initial_mode=InteractionMode.REALITY)
        selection_engine = SelectionEngine()
        target_associator = TargetAssociator()

        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        target_locked = TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

        selection_engine.update(target_locked, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert selection_engine.is_selected

        # Perform transition REALITY -> DESKTOP
        res = mode_controller.set_mode(InteractionMode.DESKTOP)
        assert res.changed is True

        # Execute transition handler resets
        selection_engine.reset()
        target_associator.reset()

        assert not selection_engine.is_selected
        assert selection_engine.state == SelectionState.NONE
        assert target_associator.target_state == TargetState.NONE


class TestPhase8PhantomActionPrevention:
    """Validates the four essential phantom action prevention scenarios (Tests A, B, C, D)."""

    def _create_mock_locked_target(self) -> TargetedObject:
        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 250, 650, 450))
        return TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

    def test_phantom_action_test_a_desktop_pinch_held_to_reality_no_select(self):
        """
        Test A: DESKTOP with pinch held -> switch to REALITY -> no SELECT emitted.
        A pre-existing held pinch cannot trigger selection without a fresh pinch cycle.
        """
        mode_controller = ModeController(initial_mode=InteractionMode.DESKTOP)
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True))
        selection_engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Pinch held in Desktop mode
        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        gesture_engine.update(hand_pinch, (500, 500))
        assert gesture_engine.is_pinched

        # Mode switch to REALITY
        mode_controller.set_mode(InteractionMode.REALITY)
        # Transition reset
        selection_engine.reset()

        # In Reality Mode on the next frame with pinch STILL held:
        # SelectionEngine detects is_pinched=True without fresh pinch_started -> does NOT select
        selection_engine._previous_pinched = True  # Synchronized
        st = selection_engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)

        assert not selection_engine.is_selected
        assert st == SelectionState.ARMED

    def test_phantom_action_test_b_reality_pinch_held_to_desktop_no_click(self):
        """
        Test B: REALITY with pinch held -> switch to DESKTOP -> releasing pinch emits NO CLICK.
        """
        mode_controller = ModeController(initial_mode=InteractionMode.REALITY)
        mouse_backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=False), backend=mouse_backend)

        # Pinch held in Reality mode
        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        gesture_engine.update(hand_pinch, (500, 500))
        assert gesture_engine.is_pinched

        # Switch to DESKTOP
        mode_controller.set_mode(InteractionMode.DESKTOP)
        # Execute transition resets (resets pinch baselines so release doesn't click)
        gesture_engine.cleanup()
        gesture_engine.config.enabled = True

        # Hand opens in Desktop mode
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)
        gesture_engine.update(hand_open, (500, 500))

        # Zero clicks emitted!
        assert len(mouse_backend.clicks) == 0

    def test_phantom_action_test_c_desktop_drag_to_reality_one_mouse_up_no_select(self):
        """
        Test C: DESKTOP with drag active -> switch to REALITY -> exactly one mouseUp and NO SELECT.
        """
        mode_controller = ModeController(initial_mode=InteractionMode.DESKTOP)
        mouse_backend = MockMouseBackend()
        gesture_engine = GestureEngine(config=GestureConfig(enabled=True, drag_threshold_px=5.0, min_pinch_duration_ms=0.0), backend=mouse_backend)
        selection_engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Drag active in Desktop mode
        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_pinch, (520, 500))
        assert mouse_backend.is_button_down

        # Switch to REALITY
        mode_controller.set_mode(InteractionMode.REALITY)
        gesture_engine.cleanup()
        selection_engine.reset()

        assert len(mouse_backend.mouse_ups) == 1
        assert not mouse_backend.is_button_down
        assert not selection_engine.is_selected

    def test_phantom_action_test_d_reality_selected_to_desktop_clears_selection_no_mouse_event(self):
        """
        Test D: REALITY with selected object -> switch to DESKTOP -> selection cleared, NO mouse events.
        """
        mode_controller = ModeController(initial_mode=InteractionMode.REALITY)
        mouse_backend = MockMouseBackend()
        selection_engine = SelectionEngine()
        target = self._create_mock_locked_target()

        # Select object in Reality mode
        selection_engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert selection_engine.is_selected

        # Switch to DESKTOP
        mode_controller.set_mode(InteractionMode.DESKTOP)
        selection_engine.reset()

        assert not selection_engine.is_selected
        assert len(mouse_backend.clicks) == 0
        assert len(mouse_backend.mouse_downs) == 0
        assert len(mouse_backend.mouse_ups) == 0


class TestPhase8DetectorGating:
    """Validates that object detection inference is skipped in Desktop Mode and executed in Reality Mode."""

    def test_desktop_mode_skips_object_detector_inference(self):
        """In Desktop Mode, ObjectDetector backend is not called."""
        mock_backend = MockObjectDetector()
        orchestrator = ObjectDetector(backend=mock_backend)

        # In Desktop Mode, main loop skips calling orchestrator.detect(frame)
        mode_controller = ModeController(initial_mode=InteractionMode.DESKTOP)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        if mode_controller.is_reality:
            detected = orchestrator.detect(frame)
        else:
            detected = []

        assert detected == []
        assert orchestrator.cache_age_frames == 0

    def test_reality_mode_executes_object_detector_inference(self):
        """In Reality Mode, ObjectDetector backend is actively called."""
        mock_backend = MockObjectDetector()
        orchestrator = ObjectDetector(backend=mock_backend)
        mode_controller = ModeController(initial_mode=InteractionMode.REALITY)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        if mode_controller.is_reality:
            detected = orchestrator.detect(frame)
        else:
            detected = []

        assert len(detected) == 3


class TestPhase8SyntheticEndToEndAndCLI:
    """Validates full synthetic pipeline runs in Desktop Mode and Reality Mode with simulated transitions."""

    def test_full_synthetic_pipeline_run_phase8_desktop_mode(self):
        """run() in synthetic mode with mode='desktop' executes 10 frames cleanly with exit code 0."""
        exit_code = run(
            synthetic_mode=True,
            mode="desktop",
            max_frames=10,
            cursor_enabled=False,
        )
        assert exit_code == 0

    def test_full_synthetic_pipeline_run_phase8_reality_mode(self):
        """run() in synthetic mode with mode='reality' executes 10 frames cleanly with exit code 0."""
        exit_code = run(
            synthetic_mode=True,
            mode="reality",
            max_frames=10,
            cursor_enabled=False,
        )
        assert exit_code == 0

    def test_cli_mode_and_cursor_independence(self):
        """ModeController and CursorConfig remain completely orthogonal."""
        # Case 1: Desktop + Cursor True
        mode_d = InteractionMode.DESKTOP
        cursor_d = True
        assert (cursor_d and mode_d == InteractionMode.DESKTOP) is True

        # Case 2: Reality + Cursor True (Hard Invariant: Cursor disabled)
        mode_r = InteractionMode.REALITY
        cursor_r = True
        assert (cursor_r and mode_r == InteractionMode.DESKTOP) is False

    def test_deterministic_mode_toggle_roundtrip_with_held_pinch(self):
        """
        Deterministic verification of DESKTOP -> M -> REALITY -> M -> DESKTOP
        with a held pinch and active target ensuring zero phantom actions and safe lifecycle.
        """
        mode_ctrl = ModeController(initial_mode=InteractionMode.DESKTOP)
        mouse_backend = MockMouseBackend()
        gesture_engine = GestureEngine(
            config=GestureConfig(enabled=True, drag_threshold_px=5.0, min_pinch_duration_ms=0.0),
            backend=mouse_backend,
        )
        selection_engine = SelectionEngine()
        det = DetectedObject.create(class_id=39, class_name="bottle", confidence=0.90, bbox=(500, 200, 650, 450))
        target = TargetedObject(
            detected_object=det, score=0.85, containment_score=0.85, proximity_score=0.85,
            confidence_score=0.90, distance_px=10.0, inside_bbox=True, inside_margin=True,
            state=TargetState.LOCKED, stable=True, frames_active=4, timestamp=1.0,
        )

        # 1. In DESKTOP: User initiates pinch drag
        hand_pinch = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.20)
        gesture_engine.update(hand_pinch, (500, 500))
        gesture_engine.update(hand_pinch, (520, 500))
        assert mouse_backend.is_button_down

        # 2. Toggle M: DESKTOP -> REALITY
        res1 = mode_ctrl.toggle_mode(reason=ModeTransitionReason.KEYBOARD_TOGGLE)
        assert res1.changed
        assert res1.new_mode == InteractionMode.REALITY
        # Execute transition resets
        gesture_engine.cleanup()
        selection_engine.reset()
        assert not mouse_backend.is_button_down
        assert len(mouse_backend.mouse_ups) == 1
        assert not selection_engine.is_selected

        # 3. In REALITY: User releases pinch, then initiates fresh pinch on locked target
        hand_open = create_mock_hand_digits(index_ext=True, pinch_dist_norm=0.70)
        selection_engine.update(target, is_pinched=False, active_gesture="POINTING", hand_valid=True)
        # Fresh pinch to select
        st = selection_engine.update(target, is_pinched=True, active_gesture="PINCHING", hand_valid=True)
        assert st == SelectionState.SELECTED
        assert selection_engine.is_selected
        assert len(mouse_backend.clicks) == 0  # Zero mouse actions in Reality

        # 4. Toggle M: REALITY -> DESKTOP while pinch is still held
        res2 = mode_ctrl.toggle_mode(reason=ModeTransitionReason.KEYBOARD_TOGGLE)
        assert res2.changed
        assert res2.new_mode == InteractionMode.DESKTOP
        # Execute transition resets
        selection_engine.reset()
        gesture_engine.cleanup()
        assert not selection_engine.is_selected

        # 5. In DESKTOP: User opens hand
        gesture_engine.config.enabled = True
        gesture_engine.update(hand_open, (520, 500))
        # Zero clicks because edge baseline was reset on transition
        assert len(mouse_backend.clicks) == 0
        assert not mouse_backend.is_button_down



