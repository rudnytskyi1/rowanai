"""Modern HTML/CSS HUD overlay for the room TV (self-contained, OPTIONAL).

A transparent, borderless, always-on-top, click-through, full-screen window
that gives the room TV a subtle "presence" for the assistant: a small, soft
blurred glow, centered on screen, that stays completely hidden while the
assistant is idle and only fades in while something is actually happening -
listening, thinking, speaking, a click ping, a screen scan, the typing dots,
a status caption, or a quick wake/error flash. When the activity ends it
fades back out and disappears again.

Unlike the Tkinter version this replaces, the actual look is a real HTML/CSS
page (:data:`HUD_HTML_PATH`, ``client/overlay_web/hud.html``) rendered by a
Qt ``QWebEngineView``: CSS ``radial-gradient`` + ``filter: blur()`` + real
alpha compositing give a genuinely soft, semi-transparent glow - Tk's
``-transparentcolor`` chroma key could only ever produce an opaque disc with
a hard edge, no matter how many concentric ovals were drawn on top of it.

PySide6 owns a ``QApplication`` + ``QWebEngineView`` on ONE dedicated UI
thread - like :mod:`client.viewer`'s ``cv2`` window and :mod:`client.camera`'s
capture/inference threads, Qt's own event loop is not safe to drive from more
than one thread and must never be driven from asyncio. The public methods
below only validate their argument and emit a small Qt signal, then return
immediately; a signal emitted from a thread other than the receiving
object's own thread is automatically marshaled onto that thread's event loop
by Qt (a "queued connection" - PySide6 picks this automatically whenever the
emitting and receiving threads differ, so no lock is needed here). A small
:class:`QObject` "bridge" living on the UI thread is the ONLY thing that ever
touches ``PySide6`` after construction: it drives the window's native
``show()``/``hide()`` and pushes the new state into the page with
``page().runJavaScript("window.hudState({...})")`` (and ``window.hudClick``/
``window.hudFlash`` for the one-shot effects).

The window itself is frameless, always-on-top, a ``Qt.Tool`` (no taskbar
entry, never takes focus) and click-through (``Qt.WindowTransparentForInput``),
with ``Qt.WA_TranslucentBackground`` and a transparent page background so the
desktop under it is never touched - real per-pixel alpha, not a chroma key.
It is shown only while something is happening and ``hide()``-n at rest, same
as before; the HTML side ALSO fades everything to ``opacity: 0`` at idle, so
even the instant before/after the native hide the page itself shows nothing.

Everything here is OPTIONAL and best-effort: if the window cannot be created
for ANY reason (``PySide6``/``QtWebEngine`` not installed, no display, the
page failing to load) exactly ONE warning is logged, :attr:`OverlayHUD.enabled`
flips to ``False``, and every public method silently becomes a no-op - the
voice pipeline must never notice or care that the HUD could not be shown.
Nothing at module import time touches ``PySide6`` - only
:meth:`OverlayHUD._create_qt_objects`, called from :meth:`OverlayHUD.start`'s
dedicated thread - so ``import client.overlay`` always succeeds even on a
machine where PySide6 is not installed.

Wiring (a human/another worker wires this into :mod:`client.main`)::

    overlay = OverlayHUD(_attr(cfg.client, "overlay"))
    overlay.start()
    ...
    overlay.set_state("listening")
    overlay.click_at(0.42, 0.61)
    ...
    overlay.stop()

Demo, run directly on the room PC to eyeball it on the TV without the rest
of the system::

    python -m client.overlay
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# --- state machine ------------------------------------------------------------
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
VALID_STATES = (STATE_IDLE, STATE_LISTENING, STATE_THINKING, STATE_SPEAKING)

# --- flash kinds ---------------------------------------------------------------
FLASH_WAKE = "wake"
FLASH_ERROR = "error"
#: The camera shutter, fired right AFTER a screenshot has been captured so the
#: owner sees what the assistant just did. Never before - see hide_now().
FLASH_SHOT = "shot"
_FLASH_KINDS = (FLASH_WAKE, FLASH_ERROR, FLASH_SHOT)

# --- config defaults -----------------------------------------------------------
# `position` and `chroma` are vestigial now (kept only so existing configs and
# callers never break): the glow is always centered, and transparency is real
# per-pixel alpha rendered by Qt, not a Tk `-transparentcolor` chroma key.
DEFAULT_POSITION = "bottom_right"
_POSITIONS = ("bottom_right", "bottom_left", "top_right", "top_left", "center")
DEFAULT_CHROMA = "#010101"

# --- timing / lifecycle tuning ---------------------------------------------------
START_TIMEOUT_S = 5.0
JOIN_TIMEOUT_S = 2.0
STATUS_MAX_CHARS = 180
#: Mirrors the HTML/CSS timings in hud.html - used to know when a one-shot
#: effect (click ring / flash bloom) has finished playing, purely so the
#: idle-hidden decision below can reconsider hiding the native window.
CLICK_TOTAL_S = 0.45
CLICK_CONVERGE_S = CLICK_TOTAL_S  # kept as a separate name for API stability
FLASH_DURATION_S = 0.5
#: The shutter's two layers run 420ms/520ms in hud.html; hold the window open
#: a touch past the longer one so the tail is never cut off by an idle hide.
SHOT_DURATION_S = 0.7
#: How long to wait before actually hide()-ing the native window once
#: nothing is active any more, so the CSS fade-out (~300-480ms) gets to
#: play out instead of being cut off by the window disappearing.
HIDE_DELAY_S = 0.55
#: How long a persistent badge (the missing echo canceller of ТЗ F-102, the
#: camera-off notice of ТЗ F-303) may hold the window open on its own.
#:
#: The badge itself never goes away - it rides in every state snapshot the page
#: receives, exactly as Ф-102/Ф-303 ask. What it may NOT do is keep a
#: transparent overlay on screen forever: the room's owner saw the window sit
#: over their desktop after every call, because "barge-in is off" counted as
#: activity for as long as the client ran. A badge now announces itself for
#: this long and then hands the screen back.
BADGE_HOLD_S = 6.0
#: How often the HUD re-asserts itself while a photo owns the screen. The
#: photo's own window re-pins on every pump (~60 times a second); 4 times a
#: second is enough to win, and cheap enough to leave running while it is up.
TOP_GUARD_INTERVAL_MS = 250

#: The actual page shown in the WebView (client/overlay_web/hud.html).
HUD_HTML_PATH = Path(__file__).resolve().parent / "overlay_web" / "chat.html"

__all__ = [
    "STATE_IDLE",
    "STATE_LISTENING",
    "STATE_THINKING",
    "STATE_SPEAKING",
    "VALID_STATES",
    "FLASH_WAKE",
    "FLASH_ERROR",
    "FLASH_SHOT",
    "norm_to_px",
    "OverlayHUD",
]


# ------------------------------------------------------------------------------
# defensive config reading (same pattern as client.camera._attr et al.)
# ------------------------------------------------------------------------------
def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a pydantic model / dataclass / mapping / ``None``."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        return default
    try:
        return bool(value)
    except Exception:  # pragma: no cover - defensive
        return default


# ------------------------------------------------------------------------------
# pure helpers (importable and testable without ever creating a QApplication)
# ------------------------------------------------------------------------------
def _validate_state(state: Any) -> str:
    """Normalise ``state``, raising :class:`ValueError` for anything unknown.

    Case-insensitive, strips whitespace. Never called on a hot path with
    anything the caller cannot afford to have rejected: :meth:`OverlayHUD.set_state`
    catches this and simply ignores an invalid state (never raises to the
    voice client), but the validation itself is a plain, pure function so it
    can be unit-tested on its own.
    """
    if not isinstance(state, str):
        raise ValueError(f"overlay state must be a string, got {state!r}")
    normalized = state.strip().lower()
    if normalized not in VALID_STATES:
        raise ValueError(
            f"unknown overlay state {state!r} (expected one of {VALID_STATES})"
        )
    return normalized


def _truncate_status(text: Any, limit: int = STATUS_MAX_CHARS) -> str:
    """Trim a status caption to ``limit`` characters, adding an ellipsis.

    ``None`` becomes ``""`` (clears the caption, per the public API). Never
    raises: anything is coerced through ``str()`` first.
    """
    value = "" if text is None else str(text).strip()
    limit = max(0, int(limit))
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1].rstrip() + "…"


def _clamp01(value: Any) -> float:
    number = float(value)
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def norm_to_px(x_norm: Any, y_norm: Any, width: int, height: int) -> tuple[int, int]:
    """Map normalized ``(0..1, 0..1)`` screen coordinates to clamped pixel ints.

    Out-of-range values are clamped rather than rejected (a slightly
    over/under-shooting vision-model coordinate is common and should still
    animate at the nearest edge, not raise). Kept for API stability: the
    click ring itself no longer needs this in Python - ``window.hudClick``
    places it with plain CSS ``vw``/``vh`` units from the normalized
    coordinates directly, so the page never needs to know the screen's
    pixel size either.
    """
    x = _clamp01(x_norm)
    y = _clamp01(y_norm)
    px = int(round(x * max(0, int(width) - 1)))
    py = int(round(y * max(0, int(height) - 1)))
    return px, py


def _is_active(
    state: str,
    status: str,
    scanning: bool,
    typing_on: bool,
    flash_active: bool,
    click_active: bool,
    badge_active: bool = False,
) -> bool:
    """Hidden-at-idle decision: is there anything worth showing right now?

    ``True`` whenever the assistant is doing something - a non-idle state
    (listening/thinking/speaking), a screen scan, the typing dots, a flash,
    a click ping, or a status caption. ``False`` only when all of those are
    quiet, which is when the native window is hidden. Pure and Qt-free so it
    can be unit-tested without ever creating a QApplication.

    ``badge_active`` is the time-boxed half of ``status``: the persistent
    badges (F-102's missing echo canceller, F-303's camera-off notice) are
    drawn while the window is up, and they may hold it up for
    :data:`BADGE_HOLD_S` - not for the whole life of the client.
    """
    if state != STATE_IDLE:
        return True
    return bool(scanning or typing_on or flash_active or click_active or status or badge_active)


def _badge_deadline(now: float, text: Any) -> float:
    """Until when a persistent badge keeps the HUD on screen on its own.

    ``0.0`` means "not at all": either there is no badge, or it was cleared.
    A badge that is already up is not extended by a redraw of the same text -
    the caller stores the deadline and only a *new* badge restarts the clock.
    """
    return float(now) + BADGE_HOLD_S if str(text or "").strip() else 0.0


def _visibility_now(owner: Any, now: float) -> bool:
    """The hidden-at-idle decision for one live HUD, at one instant.

    Split out of the Qt bridge so the rule is testable without a QApplication,
    and so the persistent badges cannot sneak back in as a permanent reason to
    stay on screen: what they contribute is ``_badge_until``, a deadline - not
    their text.
    """
    return _is_active(
        owner._state,
        owner._status or owner._speaker_name,
        owner._scanning,
        owner._typing_on,
        now < owner._flash_until,
        now < owner._click_until,
        now < getattr(owner, "_badge_until", 0.0),
    ) or bool(getattr(owner, "_control", ""))


class OverlayHUD:
    """Transparent click-through HUD - a small centered soft glow, hidden at rest.

    Construction never fails and never touches ``PySide6`` - only
    :meth:`start` does, off on its own dedicated Qt thread. Every public
    method is safe to call from any thread (the voice client's asyncio loop,
    a worker thread, ...): each one just validates its argument defensively
    and emits a Qt signal the UI thread's bridge object receives (Qt marshals
    the cross-thread delivery on its own), so none of them ever block.
    """

    def __init__(self, cfg: Any = None) -> None:
        self.enabled = _as_bool(_attr(cfg, "enabled", True), True)
        position = str(_attr(cfg, "position", DEFAULT_POSITION) or DEFAULT_POSITION).strip().lower()
        self.position = position if position in _POSITIONS else DEFAULT_POSITION
        # Vestigial: the glow is always centered and hidden-at-idle is
        # unconditional now. Still read and stored so existing configs and
        # callers never break.
        self.idle_hidden = _as_bool(_attr(cfg, "idle_hidden", False), False)
        scale = _as_float(_attr(cfg, "scale", 1.0), 1.0)
        self.scale = scale if scale > 0 else 1.0
        # Vestigial too (no chroma key any more - real alpha), but still
        # read defensively so an old config value can never raise here.
        self.chroma = str(_attr(cfg, "chroma", DEFAULT_CHROMA) or DEFAULT_CHROMA)

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._warned = False

        # -- Qt objects: created and only ever touched on the dedicated Qt
        # thread (_run_ui / the bridge). Written once, before _ready_event is
        # set, so any caller thread that gets past start() sees either a
        # fully-built bridge or None (disabled) - never a half-built one.
        self._app: Any = None
        self._view: Any = None
        self._bridge: Any = None

        # -- idle-hidden bookkeeping, Qt-thread only (mirrors the page's own
        # state so Python can decide when to show()/hide() the native
        # window; the actual look/animation lives entirely in hud.html/css).
        self._state = STATE_IDLE
        self._status = ""
        #: Persistent capability warning (ТЗ F-102: barge-in without AEC).
        self._warning = ""
        #: ТЗ F-303: the camera is off (privacy mode) — a permanent badge, not
        #: a passing status: the room must be able to see it at a glance.
        self._camera_off = ""
        #: ТЗ F-512: «Rowan управляет» — пока агент водит мышью и печатает,
        #: это видно на экране; пусто — агент не трогает ПК.
        self._control = ""
        #: Until when those badges may hold the window up on their own. They
        #: stay in the page's snapshot, but only this long as a reason to keep
        #: a transparent overlay over the owner's desktop (BADGE_HOLD_S).
        self._badge_until = 0.0
        #: When the follow-up window stops listening (ТЗ F-103), 0 when closed.
        self._followup_until = 0.0
        #: Name currently shown centred on screen (v1.7), "" when none.
        self._speaker_name = ""
        self._scanning = False
        self._typing_on = False
        self._flash_until = 0.0
        self._click_until = 0.0
        self._mapped = False
        #: Native handle of this window, so the detections photo can ask to be
        #: placed directly BELOW it instead of racing it for the top slot.
        self._hwnd: int | None = None
        self._capture_suspended = False
        #: Until when the HUD re-asserts itself above other topmost windows.
        #: The detections photo re-pins itself on every pump, so a HUD that
        #: asked for topmost once ended up underneath the picture it annotates.
        self._top_guard_until = 0.0
        self._chat = {"person": "", "messages": [], "question": ""}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the UI thread and open the overlay window (no-op if disabled).

        Blocks briefly (at most :data:`START_TIMEOUT_S`) for the window to
        either come up or fail, so callers can check :attr:`enabled` right
        after this returns. Any failure - PySide6/QtWebEngine missing, no
        display, the page failing to load - logs exactly ONE warning and
        flips :attr:`enabled` to ``False``; every other method then becomes
        a no-op. The window starts fully hidden: nothing appears on screen
        until an actual event (state/status/scan/click/typing/flash) makes
        it fade in.
        """
        if not self.enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready_event.clear()
            self._thread = threading.Thread(
                target=self._run_ui, name="jarvis-overlay-ui", daemon=True
            )
            self._thread.start()
        if not self._ready_event.wait(timeout=START_TIMEOUT_S):
            self._disable("the overlay window did not start in time")

    def stop(self) -> None:
        """Stop the UI thread and close the window (safe to call twice, or never-started)."""
        if not self.enabled:
            return
        with self._lock:
            thread = self._thread
            self._thread = None
            if thread is None or not thread.is_alive():
                return
            bridge = self._bridge
        if bridge is not None:
            try:
                bridge.stop_requested.emit()
            except Exception:  # noqa: BLE001 - a gone/closing bridge must not hang stop()
                log.debug("overlay: could not request stop", exc_info=True)
        thread.join(timeout=JOIN_TIMEOUT_S)

    def _disable(self, reason: str) -> None:
        """Flip :attr:`enabled` off, logging exactly one warning (like client.camera._fail)."""
        self.enabled = False
        if not self._warned:
            self._warned = True
            log.warning("Overlay HUD disabled: %s. The voice client keeps working normally.", reason)
        else:  # pragma: no cover - a second failure after the first warning
            log.debug("Overlay HUD failure after it was already disabled: %s", reason)

    # ------------------------------------------------------------------
    # public API - thread-safe, never blocks, never raises
    # ------------------------------------------------------------------
    def set_state(self, state: Any) -> None:
        """Switch the glow's animation: one of :data:`VALID_STATES`.

        An unknown state is logged at DEBUG and ignored - never raised to the
        caller (see :func:`_validate_state` for the pure validation logic).
        """
        if not self.enabled:
            return
        try:
            normalized = _validate_state(state)
        except ValueError:
            log.debug("Ignoring unknown overlay state %r", state)
            return
        self._post(lambda bridge: bridge.state_changed.emit(normalized))

    def set_status(self, text: Any) -> None:
        """Short caption shown just below the glow; ``""`` (or ``None``) clears it."""
        if not self.enabled:
            return
        truncated = _truncate_status(text)
        self._post(lambda bridge: bridge.status_changed.emit(truncated))

    def barge_in_warning(self, text: Any) -> None:
        """A capability warning that STAYS on the HUD (ТЗ F-102).

        ``set_status`` is a caption: it belongs to a moment and fades away with
        it. The missing echo canceller is not a moment — it is why the room
        cannot interrupt Rowan, and the owner has to be able to see it. So the
        line rides in every state snapshot the page receives and stays until it
        is cleared with an empty string.
        """
        if not self.enabled:
            return
        self._warning = _truncate_status(text)
        self._post(lambda bridge: bridge.warning_changed.emit(self._warning))

    def camera_privacy(self, text: Any) -> None:
        """The «camera off» badge of ТЗ F-303; an empty string clears it.

        Privacy is not a moment and not a status: while it lasts the room is
        not being watched, and that has to stay visible. The badge therefore
        rides in every state snapshot next to the capability warning.
        """
        if not self.enabled:
            return
        self._camera_off = _truncate_status(text)
        self._post(lambda bridge: bridge.camera_changed.emit(self._camera_off))

    def control(self, text: Any) -> None:
        """ТЗ F-512: the «Rowan is in control» badge; ``""`` clears it.

        Unlike the camera-off badge, this one is NOT time-boxed: while the agent
        drives the mouse and the keyboard the badge must stay up, and the run is
        bounded by its own 15-step limit anyway. An empty string takes it down
        the instant the run ends or the person says «стоп».
        """
        if not self.enabled:
            return
        self._control = _truncate_status(text)
        self._post(lambda bridge: bridge.control_changed.emit(self._control))

    def followup_window(self, seconds: Any) -> None:
        """Show the follow-up window draining (ТЗ F-103); ``0`` closes it.

        The window is audible only because the room knows about it: people say
        "why did it stop listening?" when nothing on screen changes. So the
        HUD draws the remaining seconds as a small ring under the glow, and
        the client closes it the moment the window stops listening - the
        indicator never outlives the microphone.
        """
        if not self.enabled:
            return
        try:
            window = max(0.0, min(30.0, float(seconds)))
        except (TypeError, ValueError):
            log.debug("Ignoring a follow-up window of %r", seconds)
            return
        self._followup_until = time.monotonic() + window if window else 0.0
        self._post(lambda bridge: bridge.followup_changed.emit(window))

    def scan_screen(self, on: bool = True) -> None:
        """Toggle the soft screen-scan sweep (while a screenshot/vision call is in flight)."""
        if not self.enabled:
            return
        self._post(lambda bridge: bridge.scan_changed.emit(bool(on)))

    def click_at(self, x_norm: Any, y_norm: Any) -> None:
        """Animate a thin ring contracting onto ``(x_norm, y_norm)`` and fading out.

        Coordinates are normalized 0..1 (clamped, never raise on an
        out-of-range or slightly malformed value); non-numeric input is
        logged at DEBUG and dropped.
        """
        if not self.enabled:
            return
        try:
            x = _clamp01(x_norm)
            y = _clamp01(y_norm)
        except (TypeError, ValueError):
            log.debug("Ignoring click_at with non-numeric coordinates: %r, %r", x_norm, y_norm)
            return
        self._post(lambda bridge: bridge.click_requested.emit(x, y))

    def typing(self, on: bool = True) -> None:
        """Toggle the three small pulsing "typing" dots below the glow."""
        if not self.enabled:
            return
        self._post(lambda bridge: bridge.typing_changed.emit(bool(on)))

    def speaker(self, name: Any) -> None:
        """Show the recognised person's NAME centred on screen; ``""`` clears it.

        Deliberately separate from :meth:`set_status`: the status caption rides
        with the glow, which tucks into the corner while Rowan works, and the
        owner asked for the name in the middle of the screen where he is
        actually looking.
        """
        if not self.enabled:
            return
        text = _truncate_status(name)
        self._post(lambda bridge: bridge.speaker_changed.emit(text))

    def flash(self, kind: str) -> None:
        """A quick accent flash: :data:`FLASH_WAKE`, :data:`FLASH_ERROR` or :data:`FLASH_SHOT`."""
        if not self.enabled:
            return
        normalized = str(kind or "").strip().lower()
        if normalized not in _FLASH_KINDS:
            log.debug("Ignoring unknown overlay flash kind %r", kind)
            return
        self._post(lambda bridge: bridge.flash_requested.emit(normalized))

    def chat(self, payload: dict) -> None:
        self._post(lambda bridge: bridge.chat_changed.emit(json.dumps(payload)))

    def chat_reply(self, text: str, sentence: str = "") -> None:
        self._post(lambda bridge: bridge.reply_changed.emit(json.dumps({"text": text, "sentence": sentence})))

    def transcript(self, payload: dict) -> None:
        self._post(lambda bridge: bridge.transcript_changed.emit(json.dumps(payload)))

    def hub_state(self, payload: dict) -> None:
        """ТЗ F-708: статус хаба (online / queue / offline) на экране комнаты."""
        self._post(lambda bridge: bridge.hub_changed.emit(json.dumps(payload)))

    def tracks(self, payload: Any) -> None:
        """ТЗ F-708: имена над треками поверх показанного кадра.

        ``payload`` — либо список ``[{name, box}]``, либо объект
        ``{"tracks": [...], "w": int, "h": int}`` с размерами самого кадра:
        полноэкранный снимок держит пропорции, и по этим числам страница сама
        считает, где именно лежит кадр, чтобы подписи встали на людей.
        """
        self._post(lambda bridge: bridge.tracks_changed.emit(
            json.dumps(payload if payload is not None else [])))

    def photo(self, jpeg: bytes) -> None:
        import base64
        self._post(lambda bridge: bridge.photo_changed.emit('data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii')))

    def confirm_voice(self, name, callback):
        """Physical confirmation on the room display, outside the chat/LLM tools."""
        if not self.enabled or self._bridge is None:
            callback(False)
            return
        cancelled = threading.Event()
        self._voice_confirm_cancel = cancelled
        self._post(lambda bridge: bridge.voice_confirmation.emit((name, callback, cancelled)))

    def cancel_voice_confirmation(self):
        cancelled = getattr(self, '_voice_confirm_cancel', None)
        if cancelled:
            cancelled.set()
        self._post(lambda bridge: bridge.voice_confirmation_cancel.emit())

    def suspend_capture(self, timeout: float = 2.0) -> bool:
        """Wait until Qt actually hides; fail closed if the UI thread is stuck."""
        cancelled = getattr(self, '_voice_confirm_cancel', None)
        if cancelled and not cancelled.is_set():
            return False
        if not self.enabled:
            return True
        if self._bridge is None:
            return False
        done = threading.Event()
        self._post(lambda bridge: bridge.capture_requested.emit(done))
        return done.wait(timeout)

    def resume_capture(self) -> None:
        self._post(lambda bridge: bridge.capture_finished.emit())

    def keep_on_top(self, seconds: float = 0.0) -> None:
        """Put the HUD back above other always-on-top windows.

        The detections photo (``client/viewer.py``) re-pins itself with
        ``SetWindowPos(HWND_TOPMOST)`` on every message-loop pump, so a window
        that only asked for topmost once at show time is pushed under the
        picture the HUD is supposed to annotate. ``seconds`` keeps the HUD
        raising itself for as long as that picture is on screen (0 = once):
        the raise never activates the window and never takes the keyboard.
        """
        if not self.enabled:
            return
        try:
            window = max(0.0, float(seconds))
        except (TypeError, ValueError):
            window = 0.0
        self._top_guard_until = max(self._top_guard_until, time.monotonic() + window)
        self._post(lambda bridge: bridge.raise_requested.emit())

    def window_handle(self) -> int | None:
        """Native handle of the HUD window, or ``None`` while it has none.

        The detections photo (``client/viewer.py``) is always-on-top too, and
        two windows that both re-assert topmost flicker while they trade the
        top slot. The photo therefore asks to be inserted directly BELOW this
        handle (``viewer.set_overlay_window``) instead of racing it. Safe to
        call from any thread: it only reads an int stored by the Qt thread.
        """
        value = getattr(self, "_hwnd", None)
        try:
            handle = int(value) if value else 0
        except (TypeError, ValueError):
            return None
        return handle or None

    def hide_now(self) -> None:
        """Take the window off screen immediately, skipping the fade-out delay.

        Everything else hides lazily (:data:`HIDE_DELAY_S`) so the CSS fade
        gets to play. That is exactly wrong right before a screen capture: the
        HUD would be baked into the screenshot the assistant is about to look
        at. This drops the native window on the spot instead. The page keeps
        its state, so the very next event (a flash, a state change) brings the
        window straight back with the glow already in the right place.
        """
        if not self.enabled:
            return
        self._post(lambda bridge: bridge.hide_now_requested.emit())

    def _post(self, fn: Any) -> None:
        """Hand an already-validated event to the bridge; never blocks, never raises.

        ``fn`` takes the bridge and emits exactly one of its Qt signals - Qt
        marshals the actual cross-thread delivery by itself, so this never
        touches the UI thread directly.
        """
        bridge = self._bridge
        if bridge is None:
            return
        try:
            fn(bridge)
        except Exception:  # noqa: BLE001 - a gone/closing bridge must not break the caller
            log.debug("Dropping an overlay event - the bridge could not accept it")

    # ------------------------------------------------------------------
    # the dedicated Qt thread
    # ------------------------------------------------------------------
    def _run_ui(self) -> None:
        """Thread entry point: build the window, then run Qt's own event loop."""
        try:
            app, view, bridge = self._create_qt_objects()
        except Exception as exc:  # noqa: BLE001 - PySide6 missing / no display / bad page
            self._disable(f"could not create the overlay window ({exc})")
            self._ready_event.set()
            return
        self._app, self._view, self._bridge = app, view, bridge
        self._ready_event.set()
        try:
            app.exec()
        except Exception as exc:  # noqa: BLE001 - the UI thread must never crash the process
            self._disable(f"the overlay UI loop failed ({exc})")
        finally:
            self._teardown_qt()

    def _create_qt_objects(self) -> tuple[Any, Any, Any]:
        """Import PySide6 and build the QApplication + transparent WebView + bridge.

        Only ever called from :meth:`_run_ui` on the dedicated Qt thread,
        which treats ANY exception here (PySide6/QtWebEngine not installed,
        no display, the page failing to construct, ...) as fatal for the
        whole HUD - a half-working window (possibly one that does not
        actually click through) is just as useless as no window at all.
        This is the ONLY place in the module that imports PySide6, so
        ``client.overlay`` stays importable on a machine without it.
        """
        # QtWebEngine must be imported before a QApplication instance is
        # created (a documented Qt/PySide6 requirement), so this import
        # happens before QApplication(...) below, not just before it is used.
        from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtWidgets import QApplication, QMessageBox

        owner = self

        class _Bridge(QObject):
            """Lives on the dedicated Qt thread; every slot below is the ONLY
            code in this module that ever touches the QWebEngineView after
            construction. Signals may be emitted from any thread - Qt
            automatically marshals a cross-thread emission into a queued
            event on this object's own (UI) thread, so nothing here needs an
            explicit lock.
            """

            state_changed = Signal(str)
            status_changed = Signal(str)
            warning_changed = Signal(str)
            camera_changed = Signal(str)
            control_changed = Signal(str)
            followup_changed = Signal(float)
            scan_changed = Signal(bool)
            typing_changed = Signal(bool)
            click_requested = Signal(float, float)
            flash_requested = Signal(str)
            speaker_changed = Signal(str)
            hide_now_requested = Signal()
            stop_requested = Signal()
            chat_changed = Signal(str)
            reply_changed = Signal(str)
            transcript_changed = Signal(str)
            hub_changed = Signal(str)
            tracks_changed = Signal(str)
            photo_changed = Signal(str)
            raise_requested = Signal()
            capture_requested = Signal(object)
            capture_finished = Signal()
            voice_confirmation = Signal(object)
            voice_confirmation_cancel = Signal()

            def __init__(self) -> None:
                super().__init__()
                self.state_changed.connect(self._on_state)
                self.status_changed.connect(self._on_status)
                self.warning_changed.connect(self._on_warning)
                self.camera_changed.connect(self._on_camera)
                self.control_changed.connect(self._on_control)
                self.followup_changed.connect(self._on_followup)
                self.scan_changed.connect(self._on_scan)
                self.typing_changed.connect(self._on_typing)
                self.click_requested.connect(self._on_click)
                self.flash_requested.connect(self._on_flash)
                self.speaker_changed.connect(self._on_speaker)
                self.hide_now_requested.connect(self._on_hide_now)
                self.stop_requested.connect(self._on_stop)
                self.chat_changed.connect(self._on_chat)
                self.reply_changed.connect(self._on_reply)
                self.transcript_changed.connect(self._on_transcript)
                self.hub_changed.connect(self._on_hub)
                self.tracks_changed.connect(self._on_tracks)
                self.photo_changed.connect(self._on_photo)
                self.raise_requested.connect(self._on_raise)
                self.capture_requested.connect(self._on_capture)
                self.capture_finished.connect(self._on_capture_finished)
                self.voice_confirmation.connect(self._on_voice_confirmation)
                self.voice_confirmation_cancel.connect(self._cancel_voice_confirmation)
                self._voice_box = None
                self._followup_timer = None

            def _cancel_voice_confirmation(self):
                if self._voice_box is not None:
                    self._voice_box.reject()

            def _on_voice_confirmation(self, request):
                name, callback, cancelled = request
                self._cancel_voice_confirmation()
                if cancelled.is_set():
                    callback(False)
                    return
                box = QMessageBox()
                self._voice_box = box
                rename = name if isinstance(name, dict) else None
                box.setWindowTitle('Rowan - confirm name change' if rename else 'Rowan - confirm voice update')
                box.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
                box.setTextFormat(Qt.PlainText)
                box.setText(f"Change {rename['old_name']} to {rename['new_name']}?" if rename else f'Update the saved voice for {name}?')
                box.setInformativeText((
                    'This combines two existing profiles, including their voices, faces, permissions and private history. '
                    'Choose Save only if both names belong to the same person.' if rename.get('merge') else
                    'Choose Save only if this corrects the name of the same person. Their voice, face, memories and history will be kept.'
                ) if rename else (
                    f'Only choose Save if {name} just read all the sentences.\n\n'
                    'The recording did not match the old profile confidently. '
                    'Saving will link this voice to that person\'s conversations and permissions.'))
                box.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
                box.setDefaultButton(QMessageBox.Cancel)
                box.setEscapeButton(QMessageBox.Cancel)
                box.setStyleSheet('QWidget { background: #18181b; color: #fafafa; font-size: 16px; } '
                                  'QPushButton { padding: 10px 24px; border: 1px solid #52525b; border-radius: 6px; }')
                def finished(_result):
                    approved = not cancelled.is_set() and box.standardButton(box.clickedButton()) == QMessageBox.Save
                    cancelled.set()
                    self._voice_box = None
                    callback(approved)
                    box.deleteLater()
                box.finished.connect(finished)
                timer = QTimer(box)
                timer.setSingleShot(True)
                timer.timeout.connect(box.reject)
                timer.start(45000)
                box.open()
                try:
                    import ctypes
                    ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.c_void_p(int(box.winId())), 0x11)
                except (AttributeError, OSError):
                    pass  # Captures also fail closed while this dialog is open.

            # -- helpers (Qt thread only) --
            def _run_js(self, script: str) -> None:
                view = owner._view
                if view is None:
                    return
                try:
                    view.page().runJavaScript(script)
                except Exception:  # noqa: BLE001 - best-effort, page may be closing
                    log.debug("overlay: runJavaScript failed", exc_info=True)

            def _sync_state(self) -> None:
                payload = json.dumps(
                    {
                        "state": owner._state,
                        "status": owner._status,
                        "warning": owner._warning,
                        "camera_off": owner._camera_off,
                        "control": owner._control,
                        "typing": owner._typing_on,
                        "scan": owner._scanning,
                    }
                )
                self._run_js(f"window.hudState && window.hudState({payload});")

            def _active_now(self) -> bool:
                return _visibility_now(owner, time.monotonic())

            def _reconsider_visibility(self) -> None:
                if self._active_now():
                    self._show()
                else:
                    QTimer.singleShot(int(HIDE_DELAY_S * 1000), self._maybe_hide)

            def _maybe_hide(self) -> None:
                if not self._active_now():
                    self._hide()

            def _show(self) -> None:
                if owner._capture_suspended or owner._mapped or owner._view is None:
                    return
                try:
                    owner._view.show()
                    # Showing is what re-asserts the topmost flag; without this
                    # a photo that appeared in the meantime keeps the top slot.
                    owner._view.raise_()
                except Exception:  # noqa: BLE001 - best-effort
                    log.debug("overlay: could not show the window", exc_info=True)
                    return
                owner._mapped = True

            def _on_raise(self) -> None:
                """ТЗ F-708/F-102: the HUD wins over the photo window.

                ``raise_`` on a ``WindowStaysOnTopHint`` window is a
                ``SetWindowPos(HWND_TOPMOST)`` call, which puts this window at
                the top of the always-on-top band without touching focus. While
                a picture is on screen the owner asks for it again and again,
                because the picture re-pins itself even faster.
                """
                if owner._view is None or not owner._mapped:
                    return
                try:
                    owner._view.raise_()
                except Exception:  # noqa: BLE001 - best-effort
                    log.debug("overlay: could not raise the window", exc_info=True)
                    return
                if time.monotonic() < owner._top_guard_until:
                    QTimer.singleShot(TOP_GUARD_INTERVAL_MS, self._on_raise)

            def _hide(self) -> None:
                if not owner._mapped or owner._view is None:
                    return
                try:
                    owner._view.hide()
                except Exception:  # noqa: BLE001 - best-effort
                    log.debug("overlay: could not hide the window", exc_info=True)
                    return
                owner._mapped = False

            # -- slots --
            def _on_chat(self, payload: str) -> None:
                owner._chat = json.loads(payload)
                self._run_js(f"window.hudChat && window.hudChat({payload});")

            def _on_reply(self, payload: str) -> None:
                self._run_js(f"window.hudReply && window.hudReply({payload});")

            def _on_transcript(self, payload: str) -> None:
                self._run_js(f"window.hudTranscript && window.hudTranscript({payload});")

            def _on_hub(self, payload: str) -> None:
                self._run_js(f"window.hudHub && window.hudHub({payload});")

            def _on_tracks(self, payload: str) -> None:
                self._run_js(f"window.hudTracks && window.hudTracks({payload});")

            def _on_photo(self, url: str) -> None:
                self._run_js(f"window.hudImage && window.hudImage({json.dumps(url)});")

            def _on_capture(self, done) -> None:
                owner._capture_suspended = True
                self._hide()
                if owner._view is not None and not owner._view.isVisible():
                    done.set()

            def _on_capture_finished(self) -> None:
                owner._capture_suspended = False
                self._reconsider_visibility()

            def _on_state(self, state: str) -> None:
                owner._state = state
                self._sync_state()
                self._reconsider_visibility()

            def _on_status(self, text: str) -> None:
                owner._status = text
                self._sync_state()
                self._reconsider_visibility()

            def _on_warning(self, text: str) -> None:
                """ТЗ F-102: the badge rides along and announces itself once."""
                owner._warning = text
                owner._badge_until = _badge_deadline(time.monotonic(), text)
                self._sync_state()
                self._reconsider_visibility()
                if owner._badge_until:
                    QTimer.singleShot(int(BADGE_HOLD_S * 1000) + 50, self._maybe_hide)

            def _on_camera(self, text: str) -> None:
                """ТЗ F-303: the camera-off badge; the window is not held forever."""
                owner._camera_off = text
                owner._badge_until = _badge_deadline(time.monotonic(), text)
                self._sync_state()
                self._reconsider_visibility()
                if owner._badge_until:
                    QTimer.singleShot(int(BADGE_HOLD_S * 1000) + 50, self._maybe_hide)

            def _on_control(self, text: str) -> None:
                """ТЗ F-512: «Rowan управляет» — стоячий значок, не момент.

                Пока агент водит мышью, комната обязана это видеть; поэтому
                значок НЕ ограничен ``BADGE_HOLD_S``, как F-102/F-303, — его
                снимает конец прогона (сообщение хаба или «стоп» в комнате).
                """
                owner._control = text
                self._sync_state()
                self._reconsider_visibility()

            def _on_followup(self, seconds: float) -> None:
                """Draw the window draining, one tick per quarter second."""
                self._stop_followup_timer()
                if seconds <= 0:
                    owner._followup_until = 0.0
                    self._run_js("window.hudFollowup && window.hudFollowup(0);")
                    return
                owner._followup_until = time.monotonic() + seconds
                self._run_js(f"window.hudFollowup && window.hudFollowup({seconds:.2f});")
                timer = QTimer()
                timer.setInterval(250)
                timer.timeout.connect(self._tick_followup)
                timer.start()
                self._followup_timer = timer

            def _stop_followup_timer(self) -> None:
                timer, self._followup_timer = getattr(self, "_followup_timer", None), None
                if timer is not None:
                    try:
                        timer.stop()
                    except Exception:  # noqa: BLE001 - a closing page must not raise
                        log.debug("overlay: could not stop the follow-up timer", exc_info=True)

            def _tick_followup(self) -> None:
                left = max(0.0, owner._followup_until - time.monotonic())
                self._run_js(f"window.hudFollowup && window.hudFollowup({left:.2f});")
                if left <= 0:
                    owner._followup_until = 0.0
                    self._stop_followup_timer()

            def _on_scan(self, on: bool) -> None:
                owner._scanning = on
                self._sync_state()
                self._reconsider_visibility()

            def _on_typing(self, on: bool) -> None:
                owner._typing_on = on
                self._sync_state()
                self._reconsider_visibility()

            def _on_click(self, x: float, y: float) -> None:
                owner._click_until = time.monotonic() + CLICK_TOTAL_S
                self._run_js(f"window.hudClick && window.hudClick({x}, {y});")
                self._reconsider_visibility()
                QTimer.singleShot(int(CLICK_TOTAL_S * 1000) + 50, self._maybe_hide)

            def _on_speaker(self, name: str) -> None:
                owner._speaker_name = name
                self._run_js(
                    f"window.hudSpeaker && window.hudSpeaker({json.dumps(name)});"
                )
                self._reconsider_visibility()
                if not name:
                    QTimer.singleShot(int(HIDE_DELAY_S * 1000), self._maybe_hide)

            def _on_flash(self, kind: str) -> None:
                hold = SHOT_DURATION_S if kind == FLASH_SHOT else FLASH_DURATION_S
                owner._flash_until = time.monotonic() + hold
                # Show BEFORE firing the animation: a flash that starts while
                # the window is still hidden loses its first frames.
                self._reconsider_visibility()
                self._run_js(f"window.hudFlash && window.hudFlash({json.dumps(kind)});")
                QTimer.singleShot(int(hold * 1000) + 50, self._maybe_hide)

            def _on_hide_now(self, *_args: Any) -> None:
                # Deliberately unconditional: the caller wants the window gone
                # this instant (it is about to photograph the screen), whether
                # or not the HUD considers itself "active".
                owner._flash_until = 0.0
                owner._click_until = 0.0
                owner._badge_until = 0.0
                self._hide()

            def _on_stop(self) -> None:
                self._cancel_voice_confirmation()
                self._stop_followup_timer()
                try:
                    if owner._view is not None:
                        owner._view.close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass
                if owner._app is not None:
                    owner._app.quit()

        app = QApplication.instance()
        if app is None:
            app = QApplication(["jarvis-overlay"])
        app.setQuitOnLastWindowClosed(False)

        view = QWebEngineView()
        view.setAttribute(Qt.WA_TranslucentBackground, True)
        view.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
        )
        view.page().setBackgroundColor(Qt.transparent)
        screen = app.primaryScreen()
        if screen is not None:
            view.setGeometry(screen.geometry())
        url = QUrl.fromLocalFile(str(HUD_HTML_PATH))
        url.setQuery(f"scale={self.scale}")
        view.load(url)
        view.hide()
        try:
            owner._hwnd = int(view.winId())
        except Exception:  # noqa: BLE001 - the HUD works without a native handle
            owner._hwnd = None

        # Defense in depth: hide/ack is still mandatory on every capture.
        try:
            import ctypes
            ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.c_void_p(int(view.winId())), 0x11)
        except (AttributeError, OSError):
            log.debug("Native capture exclusion unavailable; acknowledged hide remains active")

        bridge = _Bridge()
        return app, view, bridge

    def _teardown_qt(self) -> None:
        view, self._view = self._view, None
        self._bridge = None
        self._app = None
        self._mapped = False
        if view is not None:
            try:
                view.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
            try:
                view.deleteLater()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


# ------------------------------------------------------------------------------
# __main__ demo: cycle through every state/animation so it can be eyeballed
# on the TV without the rest of the system running.
# ------------------------------------------------------------------------------
def _run_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    hud = OverlayHUD({"enabled": True})
    hud.start()
    if not hud.enabled:
        print("Overlay could not start (PySide6/QtWebEngine missing, or no display?) - demo aborted")
        return

    print(
        "Overlay demo running - watch the centre of the TV. It stays completely "
        "hidden until something happens, then fades in. Ctrl+C to stop early."
    )
    try:
        print("... hidden (idle, nothing on screen) ...")
        time.sleep(2.0)

        print("... wake flash ...")
        hud.flash(FLASH_WAKE)
        time.sleep(1.0)

        print("... listening ...")
        hud.set_state(STATE_LISTENING)
        time.sleep(1.0)

        print("... status text ...")
        hud.set_status("listening")
        time.sleep(2.0)

        print("... thinking ...")
        hud.set_state(STATE_THINKING)
        hud.set_status("thinking")
        time.sleep(2.5)

        print("... hiding for the screenshot (nothing on screen) ...")
        hud.set_status("")
        hud.hide_now()
        time.sleep(1.0)

        print("... shutter ...")
        hud.flash(FLASH_SHOT)
        time.sleep(0.4)

        print("... scan on, thinking in the bottom-right corner ...")
        hud.set_state(STATE_THINKING)
        hud.set_status("looking at the screen")
        hud.scan_screen(True)
        time.sleep(2.5)
        print("... scan off ...")
        hud.scan_screen(False)
        time.sleep(0.5)

        print("... two clicks ...")
        hud.set_status("clicking around")
        hud.click_at(0.3, 0.4)
        time.sleep(0.6)
        hud.click_at(0.7, 0.6)
        time.sleep(0.6)

        print("... typing ...")
        hud.typing(True)
        hud.set_status("typing")
        time.sleep(2.0)
        hud.typing(False)

        print("... speaking ...")
        hud.set_state(STATE_SPEAKING)
        hud.set_status("speaking")
        time.sleep(3.0)

        print("... fading out and hidden again ...")
        hud.set_state(STATE_IDLE)
        hud.set_status("")
        time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        hud.stop()
        print("Overlay demo stopped")


if __name__ == "__main__":
    _run_demo()
