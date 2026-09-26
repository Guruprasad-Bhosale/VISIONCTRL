"""
Main application executable for VISIONCTRL.

Pipeline:
Hand Tracking -> Cursor Mapping -> Gesture Engine -> 3D Spatial Ray ->
Camera Projection -> Object Detection -> Target Association -> Selection Engine -> HUD.
"""

import argparse
from collections import deque
import logging
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera_projection import CameraProjection, ProjectedRay
from cursor_controller import (
    ActiveHandSelector,
    CursorConfig,
    CursorController,
    MockMouseBackend,
    MouseBackend,
    PyAutoGUIMouseBackend,
)
from gesture_engine import GestureConfig, GestureEngine
from hand_tracker import HandData, HandTracker
from hud import draw_hand_skeleton, draw_hud
from mode_controller import InteractionMode, ModeController, ModeTransitionReason
from object_detector import (
    DetectedObject,
    DetectorStatus,
    MediaPipeObjectDetector,
    MockObjectDetector,
    ObjectDetector,
    ObjectDetectorBackend,
    ObjectDetectorConfig,
)
from pointing_estimator import PointingConfig, PointingEstimator
from selection_engine import (
    SelectedObject,
    SelectionConfig,
    SelectionEngine,
    SelectionState,
)
from target_associator import (
    TargetAssociator,
    TargetAssociatorConfig,
    TargetedObject,
    TargetState,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("VISIONCTRL.Main")


class FPSTracker:
    """Computes a stable, non-fluctuating frames-per-second measurement using a rolling window."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size)
        self._fps: float = 0.0

    def tick(self) -> float:
        """Call on every frame loop iteration. Returns smoothed FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)

        if len(self._timestamps) >= 2:
            duration = self._timestamps[-1] - self._timestamps[0]
            if duration > 0:
                self._fps = (len(self._timestamps) - 1) / duration
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def initialize_camera(
    camera_id: int = 0,
    width: int = 1280,
    height: int = 720,
) -> Optional[cv2.VideoCapture]:
    """
    Initializes and configures the webcam.

    Args:
        camera_id: System index of the camera.
        width: Desired capture width.
        height: Desired capture height.

    Returns:
        Configured cv2.VideoCapture object or None if initialization fails.
    """
    logger.info("Attempting to open camera device %d...", camera_id)

    # Try DirectShow on Windows first for fast startup, then fallback to default backend
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.info("DirectShow unavailable; falling back to default OpenCV backend...")
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error(
            "CRITICAL: Failed to open webcam at camera_id=%d. "
            "Please check if camera is connected and not in use by another application.",
            camera_id,
        )
        return None

    # Request resolution and optimized buffering
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Test reading a single frame to confirm hardware communication
    ret, test_frame = cap.read()
    if not ret or test_frame is None or test_frame.size == 0:
        logger.error(
            "CRITICAL: Camera device %d opened but failed to read initial frame.",
            camera_id,
        )
        cap.release()
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera online: %dx%d @ camera_id=%d", actual_w, actual_h, camera_id)
    return cap


def process_frame(
    frame: np.ndarray,
    tracker: HandTracker,
) -> Tuple[np.ndarray, List[HandData]]:
    """
    Mirrors frame horizontally and processes hand tracking landmarks.

    Args:
        frame: Raw BGR frame from camera.
        tracker: Active HandTracker instance.

    Returns:
        Tuple of (mirrored_frame, detected_hands_list).
    """
    # 1. Mirror horizontally for natural, intuitive interaction
    mirrored_frame = cv2.flip(frame, 1)

    # 2. Extract hand landmark data
    hand_data_list = tracker.process_frame(mirrored_frame)

    return mirrored_frame, hand_data_list


def run(
    camera_id: int = 0,
    width: int = 1280,
    height: int = 720,
    mode: str = "desktop",
    cursor_enabled: bool = False,
    x_margin: float = 0.10,
    y_margin: float = 0.10,
    smoothing: float = 0.65,
    deadzone: float = 3.0,
    max_step_px: float = 180.0,
    arming_duration_s: float = 0.35,
    pinch_start_threshold: float = 0.35,
    pinch_release_threshold: float = 0.45,
    drag_threshold_px: float = 8.0,
    click_cooldown_s: float = 0.20,
    right_click_cooldown_s: float = 0.30,
    middle_click_cooldown_s: float = 0.30,
    activation_frames: int = 4,
    release_frames: int = 3,
    ray_length_px: float = 350.0,
    ray_direction_smoothing: float = 0.55,
    ray_activate_conf: float = 0.40,
    ray_release_conf: float = 0.30,
    origin_strategy: str = "INDEX_TIP",
    detector_confidence_threshold: float = 0.25,
    detection_interval_frames: int = 3,
    max_detection_age_frames: int = 6,
    target_margin_px: float = 14.0,
    target_activation_frames: int = 3,
    target_release_frames: int = 3,
    object_model_path: str = "models/efficientdet_lite0.tflite",
    selection_enabled: bool = True,
    min_selection_score: float = 0.40,
    selection_cooldown_frames: int = 5,
    synthetic_mode: bool = False,
    max_frames: Optional[int] = None,
    model_path: str = "models/hand_landmarker.task",
    window_name: str = "VISIONCTRL",
    mouse_backend: Optional[MouseBackend] = None,
    object_detector_backend: Optional[ObjectDetectorBackend] = None,
) -> int:
    """
    Main application loop for VISIONCTRL.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    # Initialize Mode Controller
    initial_interaction_mode = InteractionMode.REALITY if mode.lower() == "reality" else InteractionMode.DESKTOP
    mode_controller = ModeController(initial_mode=initial_interaction_mode)

    print("\n" + "=" * 65)
    print("VISIONCTRL — Camera-Based Spatial Interaction System")
    print(f"Startup Mode: {mode_controller.mode.value}")
    print(f"Physical Cursor Backend: {'ENABLED' if cursor_enabled and not synthetic_mode else 'DISABLED (Telemetry Mode)'}")
    print("=" * 65)

    # 1. Initialize Hand Tracker
    try:
        tracker = HandTracker(
            model_path=model_path,
            num_hands=2,
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        logger.error("Failed to initialize HandTracker: %s", e)
        return 1

    # 2. Initialize Mouse Backend, Cursor Controller, Gesture Engine, Pointing Estimator
    if mouse_backend is None:
        if synthetic_mode:
            mouse_backend = MockMouseBackend(screen_width=1920, screen_height=1080)
            logger.info("Operating in SYNTHETIC mode. Backend: MOCK | Physical Mouse: DISABLED")
        else:
            mouse_backend = PyAutoGUIMouseBackend()

    # Desktop mode defaults
    cursor_config = CursorConfig(
        enabled=(cursor_enabled and not synthetic_mode and mode_controller.is_desktop),
        x_margin=x_margin,
        y_margin=y_margin,
        smoothing=smoothing,
        deadzone=deadzone,
        max_step_px=max_step_px,
        arming_duration_s=arming_duration_s,
    )
    cursor_controller = CursorController(config=cursor_config, backend=mouse_backend)

    gesture_config = GestureConfig(
        enabled=(cursor_enabled and not synthetic_mode and mode_controller.is_desktop),
        pinch_start_threshold=pinch_start_threshold,
        pinch_release_threshold=pinch_release_threshold,
        drag_threshold_px=drag_threshold_px,
        click_cooldown_s=click_cooldown_s,
        right_click_cooldown_s=right_click_cooldown_s,
        middle_click_cooldown_s=middle_click_cooldown_s,
        activation_frames=activation_frames,
        release_frames=release_frames,
    )
    gesture_engine = GestureEngine(config=gesture_config, backend=mouse_backend)

    pointing_config = PointingConfig(
        origin_strategy=origin_strategy,
        ray_length_units=0.50,
        direction_smoothing=ray_direction_smoothing,
        ray_activate_confidence=ray_activate_conf,
        ray_release_confidence=ray_release_conf,
        loss_grace_period_frames=3,
    )
    pointing_estimator = PointingEstimator(config=pointing_config)
    hand_selector = ActiveHandSelector(max_lost_frames=5)

    # 3. Initialize Phase 6 Object Detector and Target Associator
    if object_detector_backend is None:
        if synthetic_mode:
            detector_backend = MockObjectDetector(
                confidence_threshold=detector_confidence_threshold
            )
            logger.info("Object Detector Backend: MOCK (Deterministic Synthetic Scene)")
        else:
            detector_backend = MediaPipeObjectDetector(
                model_path=object_model_path,
                confidence_threshold=detector_confidence_threshold,
            )
            status, detail = detector_backend.get_status()
            logger.info("Object Detector Backend: MediaPipe | Status: %s (%s)", status.value, detail)
    else:
        detector_backend = object_detector_backend

    detector_config = ObjectDetectorConfig(
        confidence_threshold=detector_confidence_threshold,
        detection_interval_frames=detection_interval_frames,
        max_detection_age_frames=max_detection_age_frames,
        model_path=object_model_path,
    )
    object_detector = ObjectDetector(config=detector_config, backend=detector_backend)

    target_associator_config = TargetAssociatorConfig(
        target_margin_px=target_margin_px,
        activation_frames=target_activation_frames,
        release_frames=target_release_frames,
    )
    target_associator = TargetAssociator(config=target_associator_config)

    # 4. Initialize Phase 7 Selection Engine
    selection_config = SelectionConfig(
        require_locked_target=True,
        minimum_target_score=min_selection_score,
        selection_cooldown_frames=selection_cooldown_frames,
        release_on_target_change=True,
        cancel_on_hand_loss=True,
        cancel_on_fist=True,
        cancel_on_open_palm=True,
    )
    selection_engine = SelectionEngine(config=selection_config)

    def execute_mode_transition_resets(old_mode: InteractionMode, new_mode: InteractionMode) -> None:
        """Executes domain-specific cleanup and state resets during mode switch."""
        if old_mode == InteractionMode.DESKTOP and new_mode == InteractionMode.REALITY:
            logger.info("Executing DESKTOP -> REALITY transition resets.")
            gesture_engine.cleanup()  # Safely releases any active mouse drag / button down
            cursor_controller.reset()
            target_associator.reset()
            selection_engine.reset()
        elif old_mode == InteractionMode.REALITY and new_mode == InteractionMode.DESKTOP:
            logger.info("Executing REALITY -> DESKTOP transition resets.")
            selection_engine.reset()
            target_associator.reset()
            pointing_estimator.reset()
            gesture_engine.cleanup()

    # 5. Initialize Camera or Synthetic Source
    cap: Optional[cv2.VideoCapture] = None
    camera_status = "ONLINE"

    if synthetic_mode:
        camera_status = "SIMULATED"
    else:
        cap = initialize_camera(camera_id=camera_id, width=width, height=height)
        if cap is None:
            print("\n" + "!" * 65)
            print(f"CAMERA ERROR: Unable to access camera device index {camera_id}.")
            print("Please ensure your webcam is connected or run with --synthetic for pipeline test.")
            print("!" * 65 + "\n")
            tracker.close()
            object_detector.close()
            selection_engine.reset()
            return 1

    fps_tracker = FPSTracker(window_size=30)
    frame_count = 0
    start_time = time.perf_counter()
    transition_suppressed_this_frame = False

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception:
        pass

    logger.info("Entering vision loop. Press 'M' to toggle mode, 'Q' or 'ESC' to exit.")

    try:
        while True:
            # Check frame limit if set
            if max_frames is not None and frame_count >= max_frames:
                logger.info("Reached target frame limit (%d frames). Exiting loop.", max_frames)
                break

            # 6. Read Frame
            if synthetic_mode:
                raw_frame = np.zeros((height, width, 3), dtype=np.uint8)
                raw_frame[::40, :] = (25, 30, 35)
                raw_frame[:, ::40] = (25, 30, 35)
            else:
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    logger.warning("Failed to grab frame from camera. Exiting loop.")
                    break

            # 7. Process Hand Tracking
            frame, hands = process_frame(raw_frame, tracker)
            frame_h, frame_w = frame.shape[:2]

            # 8. Select Active Hand
            active_hand = hand_selector.select(hands)
            now_ts = time.perf_counter()
            current_mode = mode_controller.mode

            # Configure backend action gating based on authoritative interaction mode
            is_desktop = (current_mode == InteractionMode.DESKTOP)
            is_reality = (current_mode == InteractionMode.REALITY)

            # PyAutoGUI Hard Invariant: In REALITY mode, mouse actions are strictly disabled
            cursor_controller.config.enabled = (cursor_enabled and not synthetic_mode and is_desktop)
            gesture_engine.config.enabled = (cursor_enabled and not synthetic_mode and is_desktop)

            if active_hand is not None:
                if is_desktop:
                    cursor_coords = cursor_controller.update(active_hand.index_tip.x, active_hand.index_tip.y)
                    gesture_engine.update(active_hand, cursor_coords)
                    pointing_estimate = pointing_estimator.estimate(None, "UNKNOWN", now_ts)
                    projected_ray = None
                else:
                    # REALITY Mode: Spatial Ray and Selection active; zero cursor emissions
                    cursor_coords = (active_hand.index_tip.px, active_hand.index_tip.py)
                    gesture_engine.update(active_hand, cursor_coords)
                    pointing_estimate = pointing_estimator.estimate(
                        active_hand,
                        gesture_engine.active_gesture,
                        now_ts,
                    )
                    projected_ray = CameraProjection.project_ray(
                        pointing_estimate.ray,
                        frame_w,
                        frame_h,
                        visual_length_px=ray_length_px,
                    )
            else:
                cursor_controller.reset()
                gesture_engine.update(None, None)
                pointing_estimate = pointing_estimator.estimate(None, "UNKNOWN", now_ts)
                projected_ray = None

            # 9. Object Detection & Target Association Subsystems (Mode-Gated)
            if is_reality:
                # Reality Mode: Run Object Detection and Target Association
                detected_objects = object_detector.detect(frame)
                active_gesture_name = gesture_engine.active_gesture if active_hand is not None else "UNKNOWN"
                pointing_valid = pointing_estimate.pointing_valid if pointing_estimate is not None else False

                targeted_object = target_associator.update(
                    projected_ray=projected_ray,
                    detected_objects=detected_objects,
                    active_gesture=active_gesture_name,
                    pointing_valid=pointing_valid,
                    timestamp=now_ts,
                )

                # Phase 7 Selection (Protected from same-frame transition leakage)
                if selection_enabled and not transition_suppressed_this_frame:
                    selection_state = selection_engine.update(
                        targeted_object=targeted_object,
                        is_pinched=gesture_engine.is_pinched if active_hand is not None else False,
                        active_gesture=active_gesture_name,
                        hand_valid=(active_hand is not None),
                        timestamp=now_ts,
                    )
                    selected_object = selection_engine.selected_object
                else:
                    selection_engine.reset()
                    selected_object = None
                    selection_state = SelectionState.NONE
            else:
                # Desktop Mode: Detector is safely gated/skipped to optimize performance
                detected_objects = []
                targeted_object = None
                selected_object = None
                selection_state = SelectionState.NONE

            # Reset same-frame transition suppression flag
            transition_suppressed_this_frame = False

            # 10. Measure FPS
            current_fps = fps_tracker.tick()

            # 11. Render Visuals (Skeletons + Gestures + Ray + Reticle + Objects + Selection + HUD)
            draw_hand_skeleton(
                frame,
                hands,
                active_hand=active_hand,
                cursor_controller=cursor_controller if is_desktop else None,
                gesture_engine=gesture_engine,
                pointing_estimate=pointing_estimate if is_reality else None,
                projected_ray=projected_ray if is_reality else None,
                detected_objects=detected_objects if is_reality else None,
                targeted_object=targeted_object if is_reality else None,
                selected_object=selected_object if is_reality else None,
            )
            draw_hud(
                frame,
                hands,
                active_hand=active_hand,
                cursor_controller=cursor_controller,
                gesture_engine=gesture_engine,
                pointing_estimate=pointing_estimate,
                projected_ray=projected_ray,
                fps=current_fps,
                camera_status=camera_status,
                detected_objects=detected_objects,
                targeted_object=targeted_object,
                detector_status=object_detector.get_status() if is_reality else None,
                detector_fps=object_detector.detector_fps if is_reality else 0.0,
                detector_latency_ms=object_detector.last_latency_ms if is_reality else 0.0,
                selected_object=selected_object,
                selection_state=selection_state.value if hasattr(selection_state, "value") else str(selection_state),
                interaction_mode=mode_controller.mode.value,
                last_transition_timestamp=mode_controller.last_transition_timestamp,
                last_transition_prev_mode=mode_controller.last_transition.previous_mode.value if mode_controller.last_transition else None,
            )

            # 12. Display Window & Handle Keyboard Inputs (Exit + Mode Toggle)
            try:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q', 'Q', or ESC
                    logger.info("User requested exit via keypress.")
                    break
                elif key in (ord("m"), ord("M")):
                    # Toggle Mode (DESKTOP <-> REALITY)
                    res = mode_controller.toggle_mode()
                    if res.changed:
                        execute_mode_transition_resets(res.previous_mode, res.new_mode)
                        transition_suppressed_this_frame = True
            except cv2.error as e:
                logger.debug("cv2.imshow note: %s", e)

            frame_count += 1

    except KeyboardInterrupt:
        logger.info("Vision loop interrupted by KeyboardInterrupt (Ctrl+C).")
    finally:
        # 13. Guaranteed Safe Resource & Mouse Release Cleanup
        gesture_engine.cleanup()
        pointing_estimator.reset()
        target_associator.reset()
        selection_engine.reset()
        object_detector.close()
        logger.info("All subsystems cleaned up safely.")

        if cap is not None:
            cap.release()
            logger.info("Camera released.")

        tracker.close()
        logger.info("Hand tracker closed.")

        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows destroyed.")
        except Exception:
            pass

        elapsed = time.perf_counter() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print("\n" + "=" * 65)
        print(f"SESSION SUMMARY: Processed {frame_count} frames in {elapsed:.2f}s (Avg FPS: {avg_fps:.1f})")
        print("Shutdown complete.")
        print("=" * 65 + "\n")

    return 0


def main() -> None:
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="VISIONCTRL Phase 8 — Desktop Mode + Reality Mode (Unified Interaction Context)"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target capture width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target capture height in pixels (default: 720)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="desktop",
        choices=["desktop", "reality"],
        help="Initial interaction mode: 'desktop' or 'reality' (default: desktop)",
    )
    parser.add_argument(
        "--cursor",
        "--cursor-enabled",
        dest="cursor_enabled",
        action="store_true",
        default=False,
        help="Enable OS cursor movement and gesture clicks in Desktop Mode (default: False, runs in Telemetry mode)",
    )
    parser.add_argument(
        "--no-cursor",
        dest="cursor_enabled",
        action="store_false",
        help="Explicitly disable OS mouse control (runs in Telemetry mode)",
    )
    parser.add_argument(
        "--x-margin",
        type=float,
        default=0.10,
        help="Inactive horizontal margin fraction [0.0 - 0.4] (default: 0.10)",
    )
    parser.add_argument(
        "--y-margin",
        type=float,
        default=0.10,
        help="Inactive vertical margin fraction [0.0 - 0.4] (default: 0.10)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.65,
        help="EMA smoothing coefficient for cursor [0.0 - 0.95] (default: 0.65)",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=3.0,
        help="Cursor deadzone threshold in pixels (default: 3.0)",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=180.0,
        help="Maximum per-frame cursor displacement in pixels (default: 180.0)",
    )
    parser.add_argument(
        "--arming-duration",
        type=float,
        default=0.35,
        help="Hold duration in seconds to arm cursor (default: 0.35)",
    )
    parser.add_argument(
        "--pinch-start",
        type=float,
        default=0.35,
        help="Normalized pinch distance threshold to start pinch (default: 0.35)",
    )
    parser.add_argument(
        "--pinch-release",
        type=float,
        default=0.45,
        help="Normalized pinch distance threshold to release pinch (default: 0.45)",
    )
    parser.add_argument(
        "--drag-threshold",
        type=float,
        default=8.0,
        help="Pixel displacement from pinch start to initiate drag (default: 8.0)",
    )
    parser.add_argument(
        "--click-cooldown",
        type=float,
        default=0.20,
        help="Minimum interval in seconds between left clicks (default: 0.20)",
    )
    parser.add_argument(
        "--right-click-cooldown",
        type=float,
        default=0.30,
        help="Minimum interval in seconds between right clicks (default: 0.30)",
    )
    parser.add_argument(
        "--middle-click-cooldown",
        type=float,
        default=0.30,
        help="Minimum interval in seconds between middle clicks (default: 0.30)",
    )
    parser.add_argument(
        "--activation-frames",
        type=int,
        default=4,
        help="Required stable frames to commit gesture activation (default: 4)",
    )
    parser.add_argument(
        "--release-frames",
        type=int,
        default=3,
        help="Required stable frames to release active gesture (default: 3)",
    )
    parser.add_argument(
        "--ray-length",
        type=float,
        default=350.0,
        help="Projected ray visual length in pixels (default: 350.0)",
    )
    parser.add_argument(
        "--ray-smoothing",
        type=float,
        default=0.55,
        help="EMA smoothing factor alpha for 3D ray direction (default: 0.55)",
    )
    parser.add_argument(
        "--ray-activate-conf",
        type=float,
        default=0.40,
        help="Confidence threshold to activate pointing ray (default: 0.40)",
    )
    parser.add_argument(
        "--ray-release-conf",
        type=float,
        default=0.30,
        help="Confidence threshold to release pointing ray (default: 0.30)",
    )
    parser.add_argument(
        "--origin-strategy",
        type=str,
        default="INDEX_TIP",
        choices=["INDEX_TIP", "INDEX_DIP", "INDEX_PIP", "INDEX_MCP"],
        help="Landmark reference used as ray origin (default: INDEX_TIP)",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for detected objects (default: 0.25)",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=3,
        help="Run object detection every N frames (default: 3)",
    )
    parser.add_argument(
        "--target-margin",
        type=float,
        default=14.0,
        help="Target hit-test boundary margin in pixels (default: 14.0)",
    )
    parser.add_argument(
        "--object-model",
        type=str,
        default="models/efficientdet_lite0.tflite",
        help="Path to the local object detection model asset",
    )
    parser.add_argument(
        "--selection-enabled",
        action="store_true",
        default=True,
        help="Enable object selection via pinch on locked target (default: True)",
    )
    parser.add_argument(
        "--min-selection-score",
        type=float,
        default=0.40,
        help="Minimum target score required to allow pinch selection (default: 0.40)",
    )
    parser.add_argument(
        "--selection-cooldown",
        type=int,
        default=5,
        help="Cooldown in frames between successive selection events (default: 5)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run in synthetic frame mode to validate pipeline without opening camera",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (useful for automated testing)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hand_landmarker.task",
        help="Path to the hand landmarker model asset",
    )

    args = parser.parse_args()
    exit_code = run(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        mode=args.mode,
        cursor_enabled=args.cursor_enabled,
        x_margin=args.x_margin,
        y_margin=args.y_margin,
        smoothing=args.smoothing,
        deadzone=args.deadzone,
        max_step_px=args.max_step,
        arming_duration_s=args.arming_duration,
        pinch_start_threshold=args.pinch_start,
        pinch_release_threshold=args.pinch_release,
        drag_threshold_px=args.drag_threshold,
        click_cooldown_s=args.click_cooldown,
        right_click_cooldown_s=args.right_click_cooldown,
        middle_click_cooldown_s=args.middle_click_cooldown,
        activation_frames=args.activation_frames,
        release_frames=args.release_frames,
        ray_length_px=args.ray_length,
        ray_direction_smoothing=args.ray_smoothing,
        ray_activate_conf=args.ray_activate_conf,
        ray_release_conf=args.ray_release_conf,
        origin_strategy=args.origin_strategy,
        detector_confidence_threshold=args.detector_conf,
        detection_interval_frames=args.detection_interval,
        target_margin_px=args.target_margin,
        object_model_path=args.object_model,
        selection_enabled=args.selection_enabled,
        min_selection_score=args.min_selection_score,
        selection_cooldown_frames=args.selection_cooldown,
        synthetic_mode=args.synthetic,
        max_frames=args.max_frames,
        model_path=args.model_path,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

