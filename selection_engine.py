"""
Object selection and pinch interaction module for VISIONCTRL.

Handles transitions from a LOCKED target to a SELECTED object via
a deliberate pinch gesture.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Optional, Tuple

from object_detector import DetectedObject
from target_associator import TargetedObject, TargetState

logger = logging.getLogger("VISIONCTRL.SelectionEngine")


class SelectionState(str, Enum):
    """Lifecycle states of object selection."""
    NONE = "NONE"
    ARMED = "ARMED"
    SELECTED = "SELECTED"
    CANCELLED = "CANCELLED"


class SelectionEvent(str, Enum):
    """Discrete events emitted during selection lifecycle."""
    SELECT_STARTED = "SELECT_STARTED"
    SELECTED = "SELECTED"
    SELECTION_RELEASED = "SELECTION_RELEASED"
    SELECTION_CANCELLED = "SELECTION_CANCELLED"


@dataclass
class SelectedObject:
    """Immutable representation of an explicitly selected visual object."""
    detected_object: DetectedObject
    object_identity: str
    class_id: int
    class_name: str
    confidence: float
    selection_score: float
    selected_at: float
    frame_index: int
    stable_target_frames: int

    @classmethod
    def create(
        cls,
        targeted_object: TargetedObject,
        frame_index: int = 0,
        timestamp: Optional[float] = None,
    ) -> "SelectedObject":
        """Factory method to construct a SelectedObject from a LOCKED TargetedObject."""
        det = targeted_object.detected_object
        identity = f"{det.class_id}_{det.class_name}_{det.bbox[0]}_{det.bbox[1]}"
        ts = time.perf_counter() if timestamp is None else timestamp

        return cls(
            detected_object=det,
            object_identity=identity,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            selection_score=targeted_object.score,
            selected_at=ts,
            frame_index=frame_index,
            stable_target_frames=targeted_object.frames_active,
        )


@dataclass
class SelectionConfig:
    """Configuration parameters for the SelectionEngine."""
    require_locked_target: bool = True
    minimum_target_score: float = 0.40
    selection_cooldown_frames: int = 5
    release_on_target_change: bool = True
    cancel_on_hand_loss: bool = True
    cancel_on_fist: bool = True
    cancel_on_open_palm: bool = True


class SelectionEngine:
    """
    Manages the selection lifecycle for visual objects targeted by the spatial pointing ray.
    Consumes TargetedObject and stabilized pinch gesture signals.
    """

    def __init__(self, config: Optional[SelectionConfig] = None) -> None:
        self.config = config or SelectionConfig()
        self._state: SelectionState = SelectionState.NONE
        self._selected_object: Optional[SelectedObject] = None
        self._last_event: Optional[SelectionEvent] = None
        self._selection_triggered: bool = False
        self._previous_pinched: bool = False
        self._pinch_cycle_handled: bool = False
        self._cooldown_remaining: int = 0
        self._frame_counter: int = 0

    def _clear_selection(self, event: Optional[SelectionEvent] = None) -> None:
        """Clears active selection while preserving or updating last_event."""
        self._state = SelectionState.NONE
        self._selected_object = None
        self._selection_triggered = False
        self._pinch_cycle_handled = False
        if event is not None:
            self._last_event = event

    def reset(self) -> None:
        """Resets all selection state, selected object reference, and history."""
        self._clear_selection(event=None)
        self._last_event = None
        self._previous_pinched = False
        self._cooldown_remaining = 0

    @property
    def state(self) -> SelectionState:
        """Current SelectionState."""
        return self._state

    @property
    def is_selected(self) -> bool:
        """True if an object is currently selected."""
        return self._state == SelectionState.SELECTED and self._selected_object is not None

    @property
    def selected_object(self) -> Optional[SelectedObject]:
        """Currently selected object, if any."""
        return self._selected_object

    @property
    def selection_triggered(self) -> bool:
        """True only on the frame when selection was committed."""
        return self._selection_triggered

    @property
    def last_event(self) -> Optional[SelectionEvent]:
        """Most recently emitted SelectionEvent."""
        return self._last_event

    def _matches_identity(self, obj_a: DetectedObject, obj_b: DetectedObject) -> bool:
        """Checks identity continuity between selected object and incoming detection."""
        if obj_a.class_id != obj_b.class_id or obj_a.class_name != obj_b.class_name:
            return False

        cx_a, cy_a = obj_a.center
        cx_b, cy_b = obj_b.center
        dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
        return dist <= 120.0

    def update(
        self,
        targeted_object: Optional[TargetedObject],
        is_pinched: bool,
        active_gesture: str = "POINTING",
        hand_valid: bool = True,
        timestamp: Optional[float] = None,
    ) -> SelectionState:
        """
        Updates selection state for the current vision frame.

        Args:
            targeted_object: Current TargetedObject from TargetAssociator (or None).
            is_pinched: Active thumb-index pinch from GestureEngine.
            active_gesture: Current committed gesture name.
            hand_valid: Boolean indicating whether active hand landmarks are tracked.
            timestamp: Optional monotonic timestamp.

        Returns:
            Current SelectionState.
        """
        self._frame_counter += 1
        self._selection_triggered = False
        now = time.perf_counter() if timestamp is None else timestamp

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        pinch_started = is_pinched and not self._previous_pinched
        pinch_released = not is_pinched and self._previous_pinched

        if not hand_valid:
            if self.config.cancel_on_hand_loss and self.is_selected:
                logger.info("Selection cancelled: Hand tracking lost.")
                self._clear_selection(event=SelectionEvent.SELECTION_CANCELLED)
            else:
                self._clear_selection()
            self._previous_pinched = is_pinched
            return SelectionState.NONE

        if active_gesture == "FIST":
            if self.config.cancel_on_fist and self.is_selected:
                logger.info("Selection cancelled: FIST safety lock engaged.")
                self._clear_selection(event=SelectionEvent.SELECTION_CANCELLED)
            else:
                self._clear_selection()
            self._previous_pinched = is_pinched
            return SelectionState.NONE

        if active_gesture == "OPEN_PALM":
            if self.config.cancel_on_open_palm and self.is_selected:
                logger.info("Selection cancelled: OPEN_PALM gesture active.")
                self._clear_selection(event=SelectionEvent.SELECTION_CANCELLED)
            else:
                self._clear_selection()
            self._previous_pinched = is_pinched
            return SelectionState.NONE

        if pinch_released:
            self._pinch_cycle_handled = False
            if self.is_selected:
                self._last_event = SelectionEvent.SELECTION_RELEASED

        if self._selected_object is not None:
            if targeted_object is None:
                logger.info("Selection cancelled: Target lost.")
                self._clear_selection(event=SelectionEvent.SELECTION_CANCELLED)
                self._previous_pinched = is_pinched
                return SelectionState.NONE

            if not self._matches_identity(self._selected_object.detected_object, targeted_object.detected_object):
                if self.config.release_on_target_change:
                    logger.info(
                        "Selection cancelled: Target identity changed from %s to %s.",
                        self._selected_object.class_name,
                        targeted_object.detected_object.class_name,
                    )
                    self._clear_selection(event=SelectionEvent.SELECTION_CANCELLED)
                    if (
                        targeted_object.state == TargetState.LOCKED
                        and targeted_object.score >= self.config.minimum_target_score
                    ):
                        self._state = SelectionState.ARMED
                    else:
                        self._state = SelectionState.NONE
                    self._previous_pinched = is_pinched
                    return self._state

            self._selected_object = SelectedObject(
                detected_object=targeted_object.detected_object,
                object_identity=self._selected_object.object_identity,
                class_id=self._selected_object.class_id,
                class_name=self._selected_object.class_name,
                confidence=targeted_object.detected_object.confidence,
                selection_score=targeted_object.score,
                selected_at=self._selected_object.selected_at,
                frame_index=self._selected_object.frame_index,
                stable_target_frames=targeted_object.frames_active,
            )
            self._state = SelectionState.SELECTED
            self._previous_pinched = is_pinched
            return self._state

        if targeted_object is None or targeted_object.state != TargetState.LOCKED:
            self._state = SelectionState.NONE
            self._previous_pinched = is_pinched
            return self._state

        if targeted_object.score < self.config.minimum_target_score:
            self._state = SelectionState.NONE
            self._previous_pinched = is_pinched
            return self._state

        if not is_pinched:
            self._state = SelectionState.ARMED
            self._previous_pinched = is_pinched
            return self._state

        if pinch_started and self._cooldown_remaining == 0:
            self._selected_object = SelectedObject.create(
                targeted_object=targeted_object,
                frame_index=self._frame_counter,
                timestamp=now,
            )
            self._state = SelectionState.SELECTED
            self._selection_triggered = True
            self._pinch_cycle_handled = True
            self._cooldown_remaining = self.config.selection_cooldown_frames
            self._last_event = SelectionEvent.SELECTED

            logger.info(
                "Object selected: %s (id=%s, score=%.2f, conf=%.2f)",
                self._selected_object.class_name,
                self._selected_object.object_identity,
                self._selected_object.selection_score,
                self._selected_object.confidence,
            )
            self._previous_pinched = is_pinched
            return self._state

        if self._state != SelectionState.SELECTED:
            self._state = SelectionState.ARMED
        self._previous_pinched = is_pinched
        return self._state
