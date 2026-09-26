"""
Desktop cursor controller for VISIONCTRL.

Translates normalized index-finger tracking coordinates into smoothed desktop cursor movements.
Includes active-hand lock, interaction margins, arming stabilization, max-step clamping,
deadzone filtering, and OS mouse backend interfaces.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import List, Optional, Protocol, Tuple

import numpy as np

from hand_tracker import HandData

logger = logging.getLogger("VISIONCTRL.CursorController")


class ArmingState(str, Enum):
    """Cursor activation lifecycle states."""
    DISARMED = "DISARMED"
    ARMING = "ARMING"
    ARMED = "ARMED"


class MouseBackend(Protocol):
    """Interface for OS cursor and button operations."""
    def move_to(self, x: int, y: int) -> bool:
        ...

    def mouse_down(self, button: str = "left") -> bool:
        ...

    def mouse_up(self, button: str = "left") -> bool:
        ...

    def click(self, button: str = "left") -> bool:
        ...

    def right_click(self) -> bool:
        ...

    def middle_click(self) -> bool:
        ...

    def get_screen_size(self) -> Tuple[int, int]:
        ...

    def get_cursor_pos(self) -> Tuple[int, int]:
        ...


class PyAutoGUIMouseBackend:
    """Mouse backend using PyAutoGUI with zero-latency configuration and emergency failsafe."""

    def __init__(self) -> None:
        import pyautogui
        self._pyautogui = pyautogui
        self._pyautogui.PAUSE = 0
        self._pyautogui.FAILSAFE = True
        self._error_occurred: bool = False
        self.is_button_down: bool = False

    def move_to(self, x: int, y: int) -> bool:
        if self._error_occurred:
            return False
        try:
            self._pyautogui.moveTo(x, y)
            return True
        except Exception as exc:
            logger.error("PyAutoGUI cursor movement error: %s. Pausing backend.", exc)
            self._error_occurred = True
            return False

    def mouse_down(self, button: str = "left") -> bool:
        if self._error_occurred or self.is_button_down:
            return False
        try:
            self._pyautogui.mouseDown(button=button)
            self.is_button_down = True
            return True
        except Exception as exc:
            logger.error("PyAutoGUI mouseDown error: %s. Pausing backend.", exc)
            self._error_occurred = True
            return False

    def mouse_up(self, button: str = "left") -> bool:
        if not self.is_button_down:
            return True
        try:
            self._pyautogui.mouseUp(button=button)
            self.is_button_down = False
            return True
        except Exception as exc:
            logger.error("PyAutoGUI mouseUp error: %s. Pausing backend.", exc)
            self._error_occurred = True
            self.is_button_down = False
            return False

    def click(self, button: str = "left") -> bool:
        if self._error_occurred:
            return False
        try:
            self._pyautogui.click(button=button)
            return True
        except Exception as exc:
            logger.error("PyAutoGUI click error: %s. Pausing backend.", exc)
            self._error_occurred = True
            return False

    def right_click(self) -> bool:
        return self.click(button="right")

    def middle_click(self) -> bool:
        return self.click(button="middle")

    def get_screen_size(self) -> Tuple[int, int]:
        try:
            sz = self._pyautogui.size()
            return (int(sz[0]), int(sz[1]))
        except Exception as exc:
            logger.warning("Failed to query screen size via PyAutoGUI (%s), fallback 1920x1080", exc)
            return (1920, 1080)

    def get_cursor_pos(self) -> Tuple[int, int]:
        try:
            pos = self._pyautogui.position()
            return (int(pos[0]), int(pos[1]))
        except Exception:
            return (0, 0)


class MockMouseBackend:
    """Mock mouse backend for deterministic unit tests and synthetic mode."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cursor_x: int = screen_width // 2
        self.cursor_y: int = screen_height // 2
        self.move_history: List[Tuple[int, int]] = []
        self.clicks: List[Tuple[int, int]] = []
        self.right_clicks: List[Tuple[int, int]] = []
        self.middle_clicks: List[Tuple[int, int]] = []
        self.mouse_downs: List[Tuple[int, int]] = []
        self.mouse_ups: List[Tuple[int, int]] = []
        self.is_button_down: bool = False

    def move_to(self, x: int, y: int) -> bool:
        self.cursor_x = max(0, min(self.screen_width - 1, x))
        self.cursor_y = max(0, min(self.screen_height - 1, y))
        self.move_history.append((self.cursor_x, self.cursor_y))
        return True

    def mouse_down(self, button: str = "left") -> bool:
        if self.is_button_down:
            return False
        self.is_button_down = True
        self.mouse_downs.append((self.cursor_x, self.cursor_y))
        return True

    def mouse_up(self, button: str = "left") -> bool:
        if not self.is_button_down:
            return True
        self.is_button_down = False
        self.mouse_ups.append((self.cursor_x, self.cursor_y))
        return True

    def click(self, button: str = "left") -> bool:
        if button == "right":
            return self.right_click()
        elif button == "middle":
            return self.middle_click()
        self.clicks.append((self.cursor_x, self.cursor_y))
        return True

    def right_click(self) -> bool:
        self.right_clicks.append((self.cursor_x, self.cursor_y))
        return True

    def middle_click(self) -> bool:
        self.middle_clicks.append((self.cursor_x, self.cursor_y))
        return True

    def get_screen_size(self) -> Tuple[int, int]:
        return (self.screen_width, self.screen_height)

    def get_cursor_pos(self) -> Tuple[int, int]:
        return (self.cursor_x, self.cursor_y)


class ActiveHandSelector:
    """
    Maintains a stable lock on a single active hand across frames,
    preventing cursor jumps when hand detection ordering fluctuates.
    """

    def __init__(self, max_lost_frames: int = 4) -> None:
        self.max_lost_frames = max_lost_frames
        self.locked_handedness: Optional[str] = None
        self.last_center: Optional[Tuple[int, int]] = None
        self._lost_frames_count: int = 0

    def select(self, hands: List[HandData]) -> Optional[HandData]:
        """
        Selects the active hand from detected hands list.

        Args:
            hands: List of HandData detected in the current frame.

        Returns:
            Locked active HandData or None.
        """
        if not hands:
            self._lost_frames_count += 1
            if self._lost_frames_count > self.max_lost_frames:
                self.reset()
            return None

        if self.locked_handedness is None or self.last_center is None:
            primary_hand = max(hands, key=lambda h: h.score)
            self.locked_handedness = primary_hand.handedness
            self.last_center = primary_hand.center
            self._lost_frames_count = 0
            return primary_hand

        best_match: Optional[HandData] = None
        min_dist = float("inf")

        for hand in hands:
            is_same_handedness = (hand.handedness == self.locked_handedness)
            dist = math.hypot(
                hand.center[0] - self.last_center[0],
                hand.center[1] - self.last_center[1],
            )
            effective_dist = dist if is_same_handedness else (dist + 500.0)

            if effective_dist < min_dist:
                min_dist = effective_dist
                best_match = hand

        if best_match is not None:
            self.locked_handedness = best_match.handedness
            self.last_center = best_match.center
            self._lost_frames_count = 0
            return best_match

        self._lost_frames_count += 1
        if self._lost_frames_count > self.max_lost_frames:
            self.reset()
            if hands:
                return self.select(hands)
        return None

    def reset(self) -> None:
        """Clears lock state."""
        self.locked_handedness = None
        self.last_center = None
        self._lost_frames_count = 0


@dataclass
class CursorConfig:
    """Configuration parameters for cursor coordinate mapping and filtering."""
    enabled: bool = False
    x_margin: float = 0.10
    y_margin: float = 0.10
    smoothing: float = 0.65
    deadzone: float = 3.0
    max_step_px: float = 180.0
    arming_duration_s: float = 0.35
    arming_radius_norm: float = 0.04


class CursorController:
    """
    Translates normalized index-finger landmarks into smoothed screen cursor coordinates.
    Manages arming lifecycle, boundary margins, smoothing, and OS cursor commands.
    """

    def __init__(
        self,
        config: Optional[CursorConfig] = None,
        backend: Optional[MouseBackend] = None,
    ) -> None:
        self.config = config or CursorConfig()
        self.backend = backend or PyAutoGUIMouseBackend()

        self.screen_width, self.screen_height = self.backend.get_screen_size()
        logger.info(
            "CursorController initialized. Screen: %dx%d, Enabled: %s, Smoothing: %.2f",
            self.screen_width,
            self.screen_height,
            self.config.enabled,
            self.config.smoothing,
        )

        self.arming_state: ArmingState = ArmingState.DISARMED
        self.arming_progress: float = 0.0
        self._arming_start_time: Optional[float] = None
        self._arming_anchor: Optional[Tuple[float, float]] = None

        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None
        self.last_emitted_cursor: Optional[Tuple[int, int]] = None
        self.target_screen_coords: Optional[Tuple[int, int]] = None

    def map_norm_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Maps normalized camera coordinates into screen pixels with margin clamping.

        Args:
            norm_x: Normalized X coordinate [0.0, 1.0].
            norm_y: Normalized Y coordinate [0.0, 1.0].

        Returns:
            Screen pixel coordinates (screen_x, screen_y).
        """
        if not math.isfinite(norm_x) or not math.isfinite(norm_y):
            return (self.screen_width // 2, self.screen_height // 2)

        x_margin = np.clip(self.config.x_margin, 0.0, 0.45)
        y_margin = np.clip(self.config.y_margin, 0.0, 0.45)

        x_span = max(0.01, 1.0 - 2.0 * x_margin)
        y_span = max(0.01, 1.0 - 2.0 * y_margin)

        norm_x_mapped = np.clip((norm_x - x_margin) / x_span, 0.0, 1.0)
        norm_y_mapped = np.clip((norm_y - y_margin) / y_span, 0.0, 1.0)

        screen_x = int(np.clip(round(norm_x_mapped * (self.screen_width - 1)), 0, self.screen_width - 1))
        screen_y = int(np.clip(round(norm_y_mapped * (self.screen_height - 1)), 0, self.screen_height - 1))

        return (screen_x, screen_y)

    def _update_arming(self, norm_x: float, norm_y: float) -> bool:
        """
        Updates stabilization arming state machine.
        Returns True if cursor is ARMED and active.
        """
        now = time.perf_counter()
        duration = max(0.0, self.config.arming_duration_s)

        if duration <= 0.0:
            self.arming_state = ArmingState.ARMED
            self.arming_progress = 1.0
            return True

        if self.arming_state == ArmingState.DISARMED or self._arming_anchor is None:
            self.arming_state = ArmingState.ARMING
            self.arming_progress = 0.0
            self._arming_start_time = now
            self._arming_anchor = (norm_x, norm_y)
            return False

        if self.arming_state == ArmingState.ARMING:
            ax, ay = self._arming_anchor
            drift = math.hypot(norm_x - ax, norm_y - ay)

            if drift > self.config.arming_radius_norm:
                self._arming_anchor = (norm_x, norm_y)
                self._arming_start_time = now
                self.arming_progress = 0.0
                return False

            elapsed = now - (self._arming_start_time or now)
            self.arming_progress = min(1.0, elapsed / duration)

            if elapsed >= duration:
                self.arming_state = ArmingState.ARMED
                self.arming_progress = 1.0
                return True
            return False

        return True

    def update(
        self,
        norm_x: Optional[float],
        norm_y: Optional[float],
    ) -> Optional[Tuple[int, int]]:
        """
        Processes a single frame's index-finger landmark coordinates.

        Args:
            norm_x: Normalized X coordinate from hand tracker (or None).
            norm_y: Normalized Y coordinate from hand tracker (or None).

        Returns:
            Calculated screen cursor coordinates (x, y) or None.
        """
        if norm_x is None or norm_y is None or not math.isfinite(norm_x) or not math.isfinite(norm_y):
            self.reset()
            return None

        target_x, target_y = self.map_norm_to_screen(norm_x, norm_y)
        self.target_screen_coords = (target_x, target_y)

        is_armed = self._update_arming(norm_x, norm_y)
        if not is_armed:
            return (target_x, target_y)

        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x = float(target_x)
            self.smoothed_y = float(target_y)
            self.last_emitted_cursor = (target_x, target_y)
            if self.config.enabled:
                self.backend.move_to(target_x, target_y)
            return (target_x, target_y)

        dx = target_x - self.smoothed_x
        dy = target_y - self.smoothed_y
        step_dist = math.hypot(dx, dy)

        if step_dist > self.config.max_step_px and step_dist > 0:
            scale = self.config.max_step_px / step_dist
            effective_target_x = self.smoothed_x + dx * scale
            effective_target_y = self.smoothed_y + dy * scale
        else:
            effective_target_x = float(target_x)
            effective_target_y = float(target_y)

        smoothing = np.clip(self.config.smoothing, 0.0, 0.95)
        self.smoothed_x = self.smoothed_x * smoothing + effective_target_x * (1.0 - smoothing)
        self.smoothed_y = self.smoothed_y * smoothing + effective_target_y * (1.0 - smoothing)

        out_x = int(np.clip(round(self.smoothed_x), 0, self.screen_width - 1))
        out_y = int(np.clip(round(self.smoothed_y), 0, self.screen_height - 1))

        if self.last_emitted_cursor is None:
            move_dist = float("inf")
        else:
            move_dist = math.hypot(out_x - self.last_emitted_cursor[0], out_y - self.last_emitted_cursor[1])

        if move_dist >= self.config.deadzone:
            if self.config.enabled:
                self.backend.move_to(out_x, out_y)
            self.last_emitted_cursor = (out_x, out_y)

        return (out_x, out_y)

    def reset(self) -> None:
        """Resets arming lifecycle and position filters."""
        self.arming_state = ArmingState.DISARMED
        self.arming_progress = 0.0
        self._arming_start_time = None
        self._arming_anchor = None
        self.smoothed_x = None
        self.smoothed_y = None
        self.last_emitted_cursor = None
        self.target_screen_coords = None
