"""
Interaction mode controller for VISIONCTRL.

Coordinates interaction state between DESKTOP mode (cursor mapping, gesture clicks)
and REALITY mode (spatial pointing ray, object detection, pinch selection).
"""

from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Optional

logger = logging.getLogger("VISIONCTRL.ModeController")


class InteractionMode(str, Enum):
    """Interaction modes supported by VISIONCTRL."""
    DESKTOP = "DESKTOP"
    REALITY = "REALITY"


class ModeTransitionReason(str, Enum):
    """Reasons triggering a mode transition."""
    STARTUP = "STARTUP"
    KEYBOARD_TOGGLE = "KEYBOARD_TOGGLE"
    API_REQUEST = "API_REQUEST"
    SAFETY_RESET = "SAFETY_RESET"


@dataclass(frozen=True)
class ModeTransitionResult:
    """Immutable representation of an interaction mode transition event."""
    previous_mode: InteractionMode
    new_mode: InteractionMode
    reason: ModeTransitionReason
    changed: bool
    timestamp: float


class ModeController:
    """
    Coordinates and tracks the active interaction mode in VISIONCTRL.
    Validates mode transitions and emits structured ModeTransitionResult records.
    """

    def __init__(self, initial_mode: InteractionMode = InteractionMode.DESKTOP) -> None:
        self._mode: InteractionMode = initial_mode
        self._initial_mode: InteractionMode = initial_mode
        now = time.perf_counter()
        self._last_transition: Optional[ModeTransitionResult] = ModeTransitionResult(
            previous_mode=initial_mode,
            new_mode=initial_mode,
            reason=ModeTransitionReason.STARTUP,
            changed=False,
            timestamp=now,
        )
        self._last_transition_timestamp: float = now
        logger.info("ModeController initialized. Initial mode: %s", self._mode.value)

    @property
    def mode(self) -> InteractionMode:
        """Currently active InteractionMode."""
        return self._mode

    @property
    def is_desktop(self) -> bool:
        """True if currently operating in DESKTOP mode."""
        return self._mode == InteractionMode.DESKTOP

    @property
    def is_reality(self) -> bool:
        """True if currently operating in REALITY mode."""
        return self._mode == InteractionMode.REALITY

    @property
    def last_transition(self) -> Optional[ModeTransitionResult]:
        """Most recent ModeTransitionResult."""
        return self._last_transition

    @property
    def last_transition_timestamp(self) -> float:
        """Monotonic timestamp of the last mode transition."""
        return self._last_transition_timestamp

    def set_mode(
        self,
        target_mode: InteractionMode,
        reason: ModeTransitionReason = ModeTransitionReason.API_REQUEST,
        timestamp: Optional[float] = None,
    ) -> ModeTransitionResult:
        """
        Transition to the specified target mode.

        Args:
            target_mode: Target InteractionMode.
            reason: Reason triggering the transition.
            timestamp: Optional monotonic timestamp.

        Returns:
            ModeTransitionResult describing the transition.
        """
        now = time.perf_counter() if timestamp is None else timestamp
        previous = self._mode
        changed = (target_mode != previous)

        if changed:
            self._mode = target_mode
            self._last_transition_timestamp = now
            logger.info(
                "Mode transition: %s -> %s (Reason: %s)",
                previous.value,
                target_mode.value,
                reason.value,
            )

        result = ModeTransitionResult(
            previous_mode=previous,
            new_mode=self._mode,
            reason=reason,
            changed=changed,
            timestamp=now,
        )
        self._last_transition = result
        return result

    def toggle_mode(
        self,
        reason: ModeTransitionReason = ModeTransitionReason.KEYBOARD_TOGGLE,
        timestamp: Optional[float] = None,
    ) -> ModeTransitionResult:
        """
        Toggle between DESKTOP and REALITY interaction modes.

        Args:
            reason: Reason triggering the toggle.
            timestamp: Optional monotonic timestamp.

        Returns:
            ModeTransitionResult describing the transition.
        """
        target = InteractionMode.REALITY if self._mode == InteractionMode.DESKTOP else InteractionMode.DESKTOP
        return self.set_mode(target_mode=target, reason=reason, timestamp=timestamp)

    def reset(self) -> None:
        """Reset the controller to its initial startup mode."""
        now = time.perf_counter()
        previous = self._mode
        self._mode = self._initial_mode
        self._last_transition_timestamp = now
        self._last_transition = ModeTransitionResult(
            previous_mode=previous,
            new_mode=self._initial_mode,
            reason=ModeTransitionReason.SAFETY_RESET,
            changed=(previous != self._initial_mode),
            timestamp=now,
        )
        logger.info("ModeController reset to %s", self._mode.value)
