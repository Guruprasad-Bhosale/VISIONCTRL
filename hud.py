"""
HUD and visualization module for VISIONCTRL.

Renders status overlays, gesture telemetry, spatial pointing ray,
object detection bounding boxes, target association, and selection states.
"""

from enum import Enum
import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from camera_projection import ProjectedRay
from cursor_controller import ArmingState, CursorController
from hand_tracker import HAND_CONNECTIONS, HandData, HandLandmarkIndex
from object_detector import DetectedObject, DetectorStatus
from pointing_estimator import PointingEstimate
from target_associator import TargetedObject, TargetState

# HUD Color Palette (BGR format)
COLOR_BG = (20, 24, 30)             # Dark slate background
COLOR_BORDER = (60, 70, 85)         # Subtle border
COLOR_TEXT_PRIMARY = (245, 245, 245)# Bright white
COLOR_TEXT_MUTED = (160, 170, 180)  # Muted silver
COLOR_CYAN = (240, 210, 0)          # Cyan (BGR)
COLOR_GREEN = (80, 220, 100)        # Emerald Green
COLOR_ORANGE = (0, 165, 255)        # Vibrant Amber/Orange
COLOR_YELLOW = (0, 220, 255)        # Bright Yellow
COLOR_RED = (60, 70, 240)           # Coral Red
COLOR_PURPLE = (220, 120, 180)      # Soft Purple
COLOR_MUTED_BOX = (80, 95, 110)     # Muted detection box color
COLOR_SELECTED = (240, 230, 40)     # Electric Cyan/Gold for Selected Object

# Skeleton styling
COLOR_BONE = (200, 200, 200)        # Neutral bone connection
COLOR_JOINT = (180, 180, 180)       # Standard joint node


def draw_detected_objects(
    frame: np.ndarray,
    detected_objects: Optional[List[DetectedObject]],
    targeted_object: Optional[TargetedObject] = None,
    projected_ray: Optional[ProjectedRay] = None,
    selected_object: Optional[object] = None,
) -> np.ndarray:
    """
    Renders visual bounding boxes, class labels, target lock highlights, and selection indicators.
    """
    if not detected_objects:
        return frame

    targeted_det = targeted_object.detected_object if targeted_object is not None else None
    selected_det = getattr(selected_object, "detected_object", None) if selected_object is not None else None

    for obj in detected_objects:
        x1, y1, x2, y2 = obj.bbox
        is_selected = (selected_det is not None and obj.bbox == selected_det.bbox and obj.class_name == selected_det.class_name)
        is_targeted = (targeted_det is not None and obj.bbox == targeted_det.bbox and obj.class_name == targeted_det.class_name)

        if is_selected:
            # 1. SELECTED Object Rendering (Double outline, glowing corner accents, reticle crosshair)
            sel_color = COLOR_GREEN
            cv2.rectangle(frame, (x1, y1), (x2, y2), sel_color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), COLOR_CYAN, 1, cv2.LINE_AA)

            # Heavy corner brackets
            c_len = min(22, max(8, (x2 - x1) // 3), max(8, (y2 - y1) // 3))
            # Top-Left
            cv2.line(frame, (x1 - 3, y1 - 3), (x1 + c_len, y1 - 3), COLOR_CYAN, 3, cv2.LINE_AA)
            cv2.line(frame, (x1 - 3, y1 - 3), (x1 - 3, y1 + c_len), COLOR_CYAN, 3, cv2.LINE_AA)
            # Top-Right
            cv2.line(frame, (x2 + 3, y1 - 3), (x2 - c_len, y1 - 3), COLOR_CYAN, 3, cv2.LINE_AA)
            cv2.line(frame, (x2 + 3, y1 - 3), (x2 + 3, y1 + c_len), COLOR_CYAN, 3, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(frame, (x1 - 3, y2 + 3), (x1 + c_len, y2 + 3), COLOR_CYAN, 3, cv2.LINE_AA)
            cv2.line(frame, (x1 - 3, y2 + 3), (x1 - 3, y2 - c_len), COLOR_CYAN, 3, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(frame, (x2 + 3, y2 + 3), (x2 - c_len, y2 + 3), COLOR_CYAN, 3, cv2.LINE_AA)
            cv2.line(frame, (x2 + 3, y2 + 3), (x2 + 3, y2 - c_len), COLOR_CYAN, 3, cv2.LINE_AA)

            # Center target crosshair
            cx, cy = obj.center
            cv2.circle(frame, (cx, cy), 6, COLOR_CYAN, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 12, COLOR_GREEN, 2, cv2.LINE_AA)

            # Selection Badge
            sel_score = getattr(selected_object, "selection_score", 1.0)
            label = f"[SELECTED] {obj.class_name.upper()} (Score: {sel_score:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            badge_y = max(lh + 8, y1 - 8)
            cv2.rectangle(frame, (x1 - 3, badge_y - lh - 4), (x1 + lw + 6, badge_y + 2), COLOR_BG, -1)
            cv2.rectangle(frame, (x1 - 3, badge_y - lh - 4), (x1 + lw + 6, badge_y + 2), COLOR_GREEN, 1)
            cv2.putText(frame, label, (x1 + 2, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GREEN, 1, cv2.LINE_AA)

            # Connect ray target to selected object center
            if projected_ray is not None and projected_ray.visible:
                tx, ty = projected_ray.endpoint_px
                cv2.line(frame, (tx, ty), (cx, cy), COLOR_GREEN, 2, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 6, COLOR_GREEN, 1, cv2.LINE_AA)

        elif is_targeted:
            # 2. Targeted Object Rendering (LOCKED or UNSTABLE)
            is_locked = (targeted_object.state == TargetState.LOCKED)
            target_color = COLOR_GREEN if is_locked else COLOR_YELLOW
            box_thickness = 2 if is_locked else 1

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), target_color, box_thickness, cv2.LINE_AA)

            # Corner accents
            corner_len = min(18, max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4))
            # Top-Left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), target_color, 3, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), target_color, 3, cv2.LINE_AA)
            # Top-Right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), target_color, 3, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), target_color, 3, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), target_color, 3, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), target_color, 3, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), target_color, 3, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), target_color, 3, cv2.LINE_AA)

            # Object center point
            cx, cy = obj.center
            cv2.circle(frame, (cx, cy), 4, target_color, -1, cv2.LINE_AA)

            # Target Lock Badge
            state_label = "TARGET LOCKED" if is_locked else "TARGET UNSTABLE"
            label = f"[{state_label}] {obj.class_name.upper()} (score: {targeted_object.score:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            badge_y = max(lh + 6, y1 - 6)
            cv2.rectangle(frame, (x1, badge_y - lh - 4), (x1 + lw + 6, badge_y + 2), COLOR_BG, -1)
            cv2.rectangle(frame, (x1, badge_y - lh - 4), (x1 + lw + 6, badge_y + 2), target_color, 1)
            cv2.putText(frame, label, (x1 + 3, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, target_color, 1, cv2.LINE_AA)

            # Connect ray target to object center with targeting vector
            if projected_ray is not None and projected_ray.visible:
                tx, ty = projected_ray.endpoint_px
                cv2.line(frame, (tx, ty), (cx, cy), target_color, 1, cv2.LINE_AA)
                cv2.circle(frame, (tx, ty), 6, target_color, 1, cv2.LINE_AA)

        else:
            # 3. Background Detected Object (Subtle/Muted outline)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_MUTED_BOX, 1, cv2.LINE_AA)
            label = f"{obj.class_name.upper()} {obj.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            badge_y = max(lh + 4, y1 - 4)
            cv2.rectangle(frame, (x1, badge_y - lh - 2), (x1 + lw + 4, badge_y + 2), COLOR_BG, -1)
            cv2.putText(frame, label, (x1 + 2, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    return frame


def draw_hud(
    frame: np.ndarray,
    hand_data_list: List[HandData],
    active_hand: Optional[HandData],
    cursor_controller: Optional[CursorController],
    gesture_engine: Optional[object] = None,
    pointing_estimate: Optional[PointingEstimate] = None,
    projected_ray: Optional[ProjectedRay] = None,
    fps: float = 0.0,
    camera_status: str = "ONLINE",
    detected_objects: Optional[List[DetectedObject]] = None,
    targeted_object: Optional[TargetedObject] = None,
    detector_status: Optional[Tuple[str, str]] = None,
    detector_fps: float = 0.0,
    detector_latency_ms: float = 0.0,
    selected_object: Optional[object] = None,
    selection_state: Optional[str] = None,
    interaction_mode: str = "DESKTOP",
    last_transition_timestamp: float = 0.0,
    last_transition_prev_mode: Optional[str] = None,
) -> np.ndarray:
    """
    Renders top-left telemetry panel and bottom instruction bar over the frame.
    Supports Phase 8 DESKTOP and REALITY interaction modes with mode-gated telemetry.
    """
    height, width = frame.shape[:2]
    is_desktop = (str(interaction_mode).upper() == "DESKTOP")
    is_reality = (str(interaction_mode).upper() == "REALITY")

    # --- TOP-LEFT TELEMETRY PANEL ---
    panel_x, panel_y = 16, 16
    panel_w, panel_h = 360, 520

    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        COLOR_BG,
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Panel border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        COLOR_BORDER,
        thickness=1,
    )

    # Title header
    cv2.putText(
        frame,
        "VISIONCTRL",
        (panel_x + 12, panel_y + 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    # Top right mode badge
    mode_badge_text = f"[{interaction_mode.upper()}]"
    mode_badge_color = COLOR_CYAN if is_desktop else COLOR_GREEN
    cv2.putText(
        frame,
        mode_badge_text,
        (panel_x + panel_w - 110, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        mode_badge_color,
        1,
        cv2.LINE_AA,
    )

    # Horizontal separator
    cv2.line(
        frame,
        (panel_x + 12, panel_y + 30),
        (panel_x + panel_w - 12, panel_y + 30),
        COLOR_BORDER,
        1,
        cv2.LINE_AA,
    )

    # Telemetry data calculations
    hand_count = len(hand_data_list)
    hand_status_str = "DETECTED" if hand_count > 0 else "NOT DETECTED"
    hand_color = COLOR_GREEN if hand_count > 0 else COLOR_TEXT_MUTED
    cam_color = COLOR_GREEN if camera_status == "ONLINE" else (COLOR_YELLOW if camera_status == "SIMULATED" else COLOR_RED)
    fps_color = COLOR_GREEN if fps >= 25 else (COLOR_YELLOW if fps >= 15 else COLOR_RED)

    # Cursor Telemetry
    if is_desktop:
        if cursor_controller is not None:
            if not cursor_controller.config.enabled:
                if cursor_controller.arming_state == ArmingState.ARMED and cursor_controller.last_emitted_cursor:
                    cx, cy = cursor_controller.last_emitted_cursor
                    cursor_status = f"TELEMETRY ({cx}, {cy})"
                    cursor_color = COLOR_YELLOW
                elif cursor_controller.arming_state == ArmingState.ARMING:
                    pct = int(cursor_controller.arming_progress * 100)
                    cursor_status = f"ARMING ({pct}%)"
                    cursor_color = COLOR_YELLOW
                else:
                    cursor_status = "TELEMETRY (DISARMED)"
                    cursor_color = COLOR_TEXT_MUTED
            else:
                if cursor_controller.arming_state == ArmingState.ARMED:
                    cx, cy = cursor_controller.last_emitted_cursor or (0, 0)
                    cursor_status = f"ACTIVE ({cx}, {cy})"
                    cursor_color = COLOR_GREEN
                elif cursor_controller.arming_state == ArmingState.ARMING:
                    pct = int(cursor_controller.arming_progress * 100)
                    cursor_status = f"ARMING ({pct}%)"
                    cursor_color = COLOR_YELLOW
                else:
                    cursor_status = "DISARMED"
                    cursor_color = COLOR_TEXT_MUTED
        else:
            cursor_status = "DISABLED"
            cursor_color = COLOR_RED
    else:
        cursor_status = "DISABLED (REALITY MODE)"
        cursor_color = COLOR_TEXT_MUTED

    # Phase 4 Gesture & Action Telemetry
    if gesture_engine is not None and active_hand is not None:
        g_type = str(getattr(gesture_engine, "active_gesture", "POINTING"))
        m_action = str(getattr(gesture_engine, "mouse_action", "IDLE")) if is_desktop else "N/A (REALITY MODE)"
        is_locked = getattr(gesture_engine, "interaction_locked", False)
        stabilizer = getattr(gesture_engine, "stabilizer", None)
        stab_str = f"{stabilizer.candidate_count}/{stabilizer.activation_frames}" if stabilizer else "1/1"

        if is_locked:
            lock_str = "ACTIVE (Fist)"
            lock_color = COLOR_RED
            gesture_color = COLOR_RED
        else:
            lock_str = "OFF"
            lock_color = COLOR_TEXT_MUTED
            if g_type in ("TWO_FINGER", "THREE_FINGER"):
                gesture_color = COLOR_PURPLE
            elif g_type == "OPEN_PALM":
                gesture_color = COLOR_CYAN
            elif g_type == "PINCHING":
                gesture_color = COLOR_GREEN
            else:
                gesture_color = COLOR_ORANGE

        action_color = COLOR_GREEN if m_action in ("CLICKED", "RIGHT_CLICKED", "MIDDLE_CLICKED", "BUTTON_DOWN") else COLOR_TEXT_MUTED
    else:
        g_type = "POINTING"
        gesture_color = COLOR_TEXT_MUTED
        m_action = "IDLE" if is_desktop else "N/A"
        action_color = COLOR_TEXT_MUTED
        lock_str = "OFF"
        lock_color = COLOR_TEXT_MUTED
        stab_str = "0/0"

    # Phase 5 Spatial Ray Telemetry
    if is_reality and pointing_estimate is not None and pointing_estimate.pointing_valid and pointing_estimate.ray.valid:
        ray_str = "ACTIVE"
        ray_color = COLOR_GREEN
        if projected_ray is not None:
            tgt_str = f"({projected_ray.endpoint_px[0]}, {projected_ray.endpoint_px[1]})"
            tgt_color = COLOR_CYAN
        else:
            tgt_str = "N/A"
            tgt_color = COLOR_TEXT_MUTED
    else:
        ray_str = "INACTIVE" if is_reality else "DISABLED (DESKTOP MODE)"
        ray_color = COLOR_TEXT_MUTED
        tgt_str = "NONE" if is_reality else "N/A"
        tgt_color = COLOR_TEXT_MUTED

    # Phase 6 Object Detector Telemetry
    if not is_reality:
        det_status_str = "DISABLED (DESKTOP MODE)"
        det_color = COLOR_TEXT_MUTED
        obj_count = 0
    else:
        if detector_status is not None:
            det_st_name, det_st_detail = detector_status
            if det_st_name == DetectorStatus.READY.value or det_st_name == "READY":
                det_status_str = f"READY ({det_st_detail})"
                det_color = COLOR_GREEN
            elif det_st_name == DetectorStatus.SIMULATED.value or det_st_name == "SIMULATED":
                det_status_str = f"SIMULATED ({det_st_detail})"
                det_color = COLOR_YELLOW
            elif det_st_name == DetectorStatus.UNAVAILABLE.value or det_st_name == "UNAVAILABLE":
                det_status_str = f"UNAVAILABLE ({det_st_detail})"
                det_color = COLOR_YELLOW
            else:
                det_status_str = f"ERROR ({det_st_detail})"
                det_color = COLOR_RED
        else:
            det_status_str = "NOT_CONFIGURED"
            det_color = COLOR_TEXT_MUTED
        obj_count = len(detected_objects) if detected_objects else 0

    # Phase 6 Target Association Telemetry
    if is_reality and targeted_object is not None:
        target_state_str = targeted_object.state.value if hasattr(targeted_object.state, "value") else str(targeted_object.state)
        target_color = COLOR_GREEN if target_state_str == "LOCKED" else COLOR_YELLOW
        target_obj_str = f"{targeted_object.detected_object.class_name.upper()} ({targeted_object.detected_object.confidence:.2f})"
        target_score_str = f"{targeted_object.score:.2f} (Dist: {int(targeted_object.distance_px)}px)"
    elif is_reality:
        if pointing_estimate is not None and pointing_estimate.pointing_valid and projected_ray is not None and projected_ray.visible:
            target_state_str = "SEARCHING"
            target_color = COLOR_YELLOW
        else:
            target_state_str = "NONE"
            target_color = COLOR_TEXT_MUTED
        target_obj_str = "NONE"
        target_score_str = "N/A"
    else:
        target_state_str = "DISABLED (DESKTOP MODE)"
        target_color = COLOR_TEXT_MUTED
        target_obj_str = "N/A"
        target_score_str = "N/A"

    # Phase 7 Selection Telemetry
    if is_reality and selected_object is not None:
        sel_st_str = "ACTIVE (SELECTED)"
        sel_color = COLOR_GREEN
        sel_obj_str = f"{selected_object.class_name.upper()} ({selected_object.confidence:.2f})"
        sel_score_str = f"{selected_object.selection_score:.2f}"
    elif is_reality:
        if selection_state == "ARMED" or (targeted_object and targeted_object.state == TargetState.LOCKED):
            sel_st_str = "ARMED (PINCH TO SELECT)"
            sel_color = COLOR_YELLOW
        else:
            sel_st_str = "NONE"
            sel_color = COLOR_TEXT_MUTED
        sel_obj_str = "NONE"
        sel_score_str = "N/A"
    else:
        sel_st_str = "DISABLED (DESKTOP MODE)"
        sel_color = COLOR_TEXT_MUTED
        sel_obj_str = "N/A"
        sel_score_str = "N/A"

    det_perf_str = f"{detector_latency_ms:.1f}ms" + (f" ({detector_fps:.1f} FPS)" if detector_fps > 0 else "") if is_reality else "OFF (GATED)"

    lines = [
        ("MODE:", interaction_mode.upper(), mode_badge_color),
        ("CAMERA:", camera_status, cam_color),
        ("FPS:", f"{fps:.1f}", fps_color),
        ("HAND:", hand_status_str, hand_color),
        ("HANDS:", f"{hand_count} (Active: {active_hand.handedness if active_hand else 'None'})", COLOR_TEXT_PRIMARY),
        ("GESTURE:", g_type, gesture_color),
        ("ACTION:", m_action, action_color),
        ("STABILITY:", stab_str, COLOR_TEXT_PRIMARY),
        ("LOCK:", lock_str, lock_color),
        ("CURSOR:", cursor_status, cursor_color),
        ("RAY:", ray_str, ray_color),
        ("RAY TARGET:", tgt_str, tgt_color),
        ("DETECTOR:", det_status_str, det_color),
        ("OBJECTS:", f"{obj_count} detected" if is_reality else "N/A", COLOR_TEXT_PRIMARY if obj_count > 0 else COLOR_TEXT_MUTED),
        ("TARGET:", target_state_str, target_color),
        ("TARGET OBJ:", target_obj_str, target_color if (targeted_object and is_reality) else COLOR_TEXT_MUTED),
        ("TARGET SCORE:", target_score_str, COLOR_CYAN if (targeted_object and is_reality) else COLOR_TEXT_MUTED),
        ("SELECTION:", sel_st_str, sel_color),
        ("SELECTED OBJ:", sel_obj_str, COLOR_GREEN if (selected_object and is_reality) else COLOR_TEXT_MUTED),
        ("SEL SCORE:", sel_score_str, COLOR_GREEN if (selected_object and is_reality) else COLOR_TEXT_MUTED),
        ("DET LATENCY:", det_perf_str, COLOR_TEXT_PRIMARY if (detector_latency_ms > 0 and is_reality) else COLOR_TEXT_MUTED),
        ("PHASE:", "PHASE 8: DESKTOP + REALITY MODE", COLOR_CYAN),
    ]

    row_y = panel_y + 45
    for label, val, val_color in lines:
        cv2.putText(
            frame,
            label,
            (panel_x + 12, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            val,
            (panel_x + 110, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            val_color,
            1,
            cv2.LINE_AA,
        )
        row_y += 18

    # --- BOTTOM COMMAND / STATUS BAR ---
    bar_h = 32
    bar_y = height - bar_h - 12
    bar_x = 16
    bar_w = width - 32

    if bar_w > 100 and bar_y > 0:
        overlay_bar = frame.copy()
        cv2.rectangle(
            overlay_bar,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            COLOR_BG,
            thickness=-1,
        )
        cv2.addWeighted(overlay_bar, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            COLOR_BORDER,
            thickness=1,
        )

        # Mode toggle & Exit hints
        cv2.putText(
            frame,
            "[M] Toggle Mode | [Q / ESC] Exit",
            (bar_x + 12, bar_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLOR_TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        # Transition Toast vs Live telemetry hint
        now = time.perf_counter()
        if (now - last_transition_timestamp) < 1.5 and last_transition_prev_mode is not None:
            status_hint = f"MODE SWITCHED: {last_transition_prev_mode} -> {interaction_mode.upper()}"
            hint_color = COLOR_CYAN
        elif active_hand is None:
            status_hint = "Waiting for hand..."
            hint_color = COLOR_TEXT_MUTED
        elif gesture_engine and getattr(gesture_engine, "interaction_locked", False):
            status_hint = "SAFETY LOCK ACTIVE (Fist) — All mouse, ray, and selection actions suppressed"
            hint_color = COLOR_RED
        elif is_desktop:
            if gesture_engine and str(getattr(gesture_engine, "active_gesture", "")) == "TWO_FINGER":
                status_hint = "TWO FINGER ACTIVE — Right Click"
                hint_color = COLOR_PURPLE
            elif gesture_engine and str(getattr(gesture_engine, "active_gesture", "")) == "THREE_FINGER":
                status_hint = "THREE FINGER ACTIVE — Left Click"
                hint_color = COLOR_PURPLE
            elif gesture_engine and str(getattr(gesture_engine, "active_gesture", "")) == "OPEN_PALM":
                status_hint = "OPEN PALM (PAUSE) — Desktop interaction paused"
                hint_color = COLOR_CYAN
            elif cursor_controller and cursor_controller.config.enabled:
                status_hint = "DESKTOP MODE (ACTIVE) — Point: Cursor | 3-Finger: Left-Click | 2-Finger: Right-Click | Drag: Pinch | Fist: Lock"
                hint_color = COLOR_GREEN
            else:
                status_hint = "DESKTOP MODE (TELEMETRY) — Run with --cursor to control OS mouse | Press 'M' for Reality"
                hint_color = COLOR_YELLOW
        else:
            # Reality Mode hints
            if gesture_engine and str(getattr(gesture_engine, "active_gesture", "")) == "OPEN_PALM":
                status_hint = "OPEN PALM (PAUSE) — Spatial ray, targeting, and selection deactivated"
                hint_color = COLOR_CYAN
            elif selected_object is not None:
                status_hint = f"OBJECT SELECTED: {selected_object.class_name.upper()} — Spatial Selection Active"
                hint_color = COLOR_GREEN
            elif targeted_object is not None and targeted_object.state == TargetState.LOCKED:
                status_hint = f"OBJECT LOCKED: {targeted_object.detected_object.class_name.upper()} — Pinch to Select"
                hint_color = COLOR_GREEN
            elif targeted_object is not None and targeted_object.state == TargetState.UNSTABLE:
                status_hint = f"OBJECT TARGET ACQUIRING: {targeted_object.detected_object.class_name.upper()}..."
                hint_color = COLOR_YELLOW
            elif pointing_estimate and pointing_estimate.pointing_valid and projected_ray and projected_ray.visible:
                status_hint = f"REALITY MODE — Spatial Ray Searching for Objects ({projected_ray.endpoint_px[0]}, {projected_ray.endpoint_px[1]})"
                hint_color = COLOR_CYAN
            else:
                status_hint = "REALITY MODE — Point at objects to acquire spatial target"
                hint_color = COLOR_CYAN

        cv2.putText(
            frame,
            status_hint,
            (bar_x + 230, bar_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            hint_color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_hand_skeleton(
    frame: np.ndarray,
    hand_data_list: List[HandData],
    active_hand: Optional[HandData] = None,
    cursor_controller: Optional[CursorController] = None,
    gesture_engine: Optional[object] = None,
    pointing_estimate: Optional[PointingEstimate] = None,
    projected_ray: Optional[ProjectedRay] = None,
    detected_objects: Optional[List[DetectedObject]] = None,
    targeted_object: Optional[TargetedObject] = None,
    selected_object: Optional[object] = None,
) -> np.ndarray:
    """
    Renders skeletal bones, joints, pinch lines, gesture beacons, projected 3D spatial ray,
    detected object overlays, and selection reticles.
    """
    # 0. Render detected object bounding boxes and selection overlay first (backdrop)
    draw_detected_objects(
        frame=frame,
        detected_objects=detected_objects,
        targeted_object=targeted_object,
        projected_ray=projected_ray,
        selected_object=selected_object,
    )

    for hand_idx, hand in enumerate(hand_data_list):
        landmarks = hand.landmarks
        is_active = (active_hand is not None and hand.landmarks == active_hand.landmarks)

        # 1. Draw skeletal connections (bones)
        bone_color = COLOR_BONE if is_active else (120, 120, 120)
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = (landmarks[start_idx].px, landmarks[start_idx].py)
            pt2 = (landmarks[end_idx].px, landmarks[end_idx].py)
            cv2.line(frame, pt1, pt2, bone_color, 2 if is_active else 1, cv2.LINE_AA)

        # 2. Draw standard intermediate joint circles
        for i, lm in enumerate(landmarks):
            if i in (
                HandLandmarkIndex.WRIST,
                HandLandmarkIndex.THUMB_TIP,
                HandLandmarkIndex.INDEX_FINGER_TIP,
                HandLandmarkIndex.MIDDLE_FINGER_TIP,
                HandLandmarkIndex.RING_FINGER_TIP,
                HandLandmarkIndex.PINKY_TIP,
            ):
                continue
            cv2.circle(frame, (lm.px, lm.py), 3, COLOR_JOINT, -1, cv2.LINE_AA)

        # 3. Highlight Landmarks
        w_pt = (hand.wrist.px, hand.wrist.py)
        cv2.circle(frame, w_pt, 6, COLOR_CYAN, -1, cv2.LINE_AA)
        cv2.circle(frame, w_pt, 8, COLOR_CYAN, 1, cv2.LINE_AA)

        th_pt = (hand.thumb_tip.px, hand.thumb_tip.py)
        idx_pt = (hand.index_tip.px, hand.index_tip.py)
        mid_pt = (hand.middle_tip.px, hand.middle_tip.py)
        ring_pt = (hand.ring_tip.px, hand.ring_tip.py)
        pky_pt = (hand.pinky_tip.px, hand.pinky_tip.py)

        cv2.circle(frame, th_pt, 5, COLOR_GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, mid_pt, 5, COLOR_YELLOW, -1, cv2.LINE_AA)
        cv2.circle(frame, ring_pt, 5, COLOR_PURPLE, -1, cv2.LINE_AA)
        cv2.circle(frame, pky_pt, 5, COLOR_CYAN, -1, cv2.LINE_AA)

        # 4. Pinch Connection Line & Indicator (Active Hand)
        if is_active and gesture_engine is not None:
            is_pinched = getattr(gesture_engine, "is_pinched", False)
            if is_pinched:
                pinch_line_color = COLOR_GREEN if selected_object is None else COLOR_CYAN
                cv2.line(frame, th_pt, idx_pt, pinch_line_color, 3, cv2.LINE_AA)
                mid_x = (th_pt[0] + idx_pt[0]) // 2
                mid_y = (th_pt[1] + idx_pt[1]) // 2
                cv2.circle(frame, (mid_x, mid_y), 4, pinch_line_color, -1, cv2.LINE_AA)

        # 5. Spatial Ray Projection & Reticle (Active Hand in POINTING mode)
        if is_active and projected_ray is not None and projected_ray.visible:
            orig_px = projected_ray.origin_px
            tgt_px = projected_ray.endpoint_px

            # Cyan projected ray line
            ray_color = COLOR_GREEN if selected_object is not None else COLOR_CYAN
            cv2.line(frame, orig_px, tgt_px, ray_color, 2, cv2.LINE_AA)

            # Origin beacon
            cv2.circle(frame, orig_px, 4, ray_color, -1, cv2.LINE_AA)
            cv2.circle(frame, orig_px, 7, ray_color, 1, cv2.LINE_AA)

            # Target Reticle at endpoint
            reticle_color = COLOR_GREEN if (selected_object or (targeted_object and targeted_object.state == TargetState.LOCKED)) else COLOR_CYAN
            cv2.circle(frame, tgt_px, 10, reticle_color, 1, cv2.LINE_AA)
            cv2.circle(frame, tgt_px, 3, reticle_color, -1, cv2.LINE_AA)

            # Reticle Crosshair Ticks
            tx, ty = tgt_px
            cv2.line(frame, (tx - 15, ty), (tx - 6, ty), reticle_color, 1, cv2.LINE_AA)
            cv2.line(frame, (tx + 6, ty), (tx + 15, ty), reticle_color, 1, cv2.LINE_AA)
            cv2.line(frame, (tx, ty - 15), (tx, ty - 6), reticle_color, 1, cv2.LINE_AA)
            cv2.line(frame, (tx, ty + 6), (tx, ty + 15), reticle_color, 1, cv2.LINE_AA)

            # Confidence / Status Badge near target
            if selected_object is not None:
                tgt_badge = f"SELECTED ({selected_object.class_name.upper()})"
            elif targeted_object is not None and targeted_object.state == TargetState.LOCKED:
                tgt_badge = f"LOCKED ({targeted_object.detected_object.class_name.upper()})"
            else:
                conf_val = pointing_estimate.confidence if pointing_estimate else 1.0
                tgt_badge = f"TARGET ({conf_val:.2f})"

            cv2.putText(
                frame,
                tgt_badge,
                (tx + 14, ty - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                reticle_color,
                1,
                cv2.LINE_AA,
            )

        # 6. Gesture-Specific Visual Overlay
        if is_active:
            g_type = str(getattr(gesture_engine, "active_gesture", "POINTING")) if gesture_engine else "POINTING"
            is_locked = getattr(gesture_engine, "interaction_locked", False) if gesture_engine else False

            if is_locked:
                pointer_color = COLOR_RED
                cv2.putText(frame, "LOCKED", (idx_pt[0] + 14, idx_pt[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_RED, 1, cv2.LINE_AA)
            elif g_type == "TWO_FINGER":
                pointer_color = COLOR_PURPLE
                cv2.circle(frame, mid_pt, 9, COLOR_PURPLE, 2, cv2.LINE_AA)
                cv2.putText(frame, "RIGHT-CLICK", (idx_pt[0] + 14, idx_pt[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_PURPLE, 1, cv2.LINE_AA)
            elif g_type == "THREE_FINGER":
                pointer_color = COLOR_PURPLE
                cv2.circle(frame, mid_pt, 9, COLOR_PURPLE, 2, cv2.LINE_AA)
                cv2.circle(frame, ring_pt, 9, COLOR_PURPLE, 2, cv2.LINE_AA)
                cv2.putText(frame, "LEFT-CLICK", (idx_pt[0] + 14, idx_pt[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_PURPLE, 1, cv2.LINE_AA)
            elif g_type == "OPEN_PALM":
                pointer_color = COLOR_CYAN
                cv2.putText(frame, "PAUSE", (idx_pt[0] + 14, idx_pt[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1, cv2.LINE_AA)
            elif getattr(gesture_engine, "is_dragging", False):
                pointer_color = COLOR_PURPLE
                cv2.putText(frame, "DRAG", (idx_pt[0] + 14, idx_pt[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_PURPLE, 1, cv2.LINE_AA)
            elif selected_object is not None:
                pointer_color = COLOR_GREEN
            elif getattr(gesture_engine, "is_pinched", False):
                pointer_color = COLOR_GREEN
            else:
                pointer_color = COLOR_ORANGE

            # Pointer Crosshairs on Index Tip
            cv2.circle(frame, idx_pt, 7, pointer_color, -1, cv2.LINE_AA)
            cv2.circle(frame, idx_pt, 12, pointer_color, 2, cv2.LINE_AA)
            cv2.line(frame, (idx_pt[0] - 16, idx_pt[1]), (idx_pt[0] - 10, idx_pt[1]), pointer_color, 2, cv2.LINE_AA)
            cv2.line(frame, (idx_pt[0] + 10, idx_pt[1]), (idx_pt[0] + 16, idx_pt[1]), pointer_color, 2, cv2.LINE_AA)
            cv2.line(frame, (idx_pt[0], idx_pt[1] - 16), (idx_pt[0], idx_pt[1] - 10), pointer_color, 2, cv2.LINE_AA)
            cv2.line(frame, (idx_pt[0], idx_pt[1] + 10), (idx_pt[0], idx_pt[1] + 16), pointer_color, 2, cv2.LINE_AA)

        else:
            cv2.circle(frame, idx_pt, 6, COLOR_ORANGE, -1, cv2.LINE_AA)
            cv2.circle(frame, idx_pt, 9, COLOR_ORANGE, 1, cv2.LINE_AA)

        # 7. Handedness label badge
        label_text = f"{hand.handedness} ({int(hand.score * 100)}%)" + (" [ACTIVE]" if is_active else "")
        label_x = max(10, hand.wrist.px - 35)
        label_y = min(frame.shape[0] - 40, hand.wrist.py + 24)

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(
            frame,
            (label_x - 3, label_y - th - 3),
            (label_x + tw + 3, label_y + 3),
            COLOR_BG,
            -1,
        )
        cv2.putText(
            frame,
            label_text,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            COLOR_CYAN if is_active else COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

    return frame
