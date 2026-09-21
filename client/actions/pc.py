"""Windows actions on the room PC — ``pc_control``, ``run_command`` and
``mouse_click`` (SPEC §5/§8).

Implemented with pycaw (master volume) and plain ctypes ``SendInput`` (media keys,
unicode typing, hotkey combos, mouse moves and clicks), plus Win32 calls for
monitor power, window minimizing and suspend. Applications are resolved through
:mod:`client.actions.apps`, which indexes everything installed (Start Menu +
Store apps) on top of the ``cfg.client.apps`` overrides.

v1.2 adds two computer-use primitives: ``pc_control`` with ``minimize_app``
(``EnumWindows`` over the app's processes, then ``ShowWindow(SW_MINIMIZE)``) and
the standalone ``mouse_click`` action the server's ``click_screen`` tool
produces — normalized 0..1 coordinates, scaled here to the real desktop
resolution so the server never has to know it.

All blocking work runs in worker threads (:func:`asyncio.to_thread`); the COM
apartment needed by pycaw is initialised inside the worker thread.
"""

from __future__ import annotations

import asyncio
import csv
import ctypes
import io
import logging
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from ctypes import wintypes
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

from .apps import (
    AppEntry,
    AppError,
    AppIndex,
    decode_console_output,
    normalize_app_name,
    powershell_executable,
)

log = logging.getLogger(__name__)

# --- pc_control command names (SPEC §5) --------------------------------------
CMD_VOLUME_SET = "volume_set"
CMD_VOLUME_UP = "volume_up"
CMD_VOLUME_DOWN = "volume_down"
CMD_MUTE = "mute"
CMD_UNMUTE = "unmute"
CMD_MEDIA_PLAY_PAUSE = "media_play_pause"
CMD_MEDIA_NEXT = "media_next"
CMD_MEDIA_PREV = "media_prev"
CMD_DISPLAY_OFF = "display_off"
CMD_DISPLAY_ON = "display_on"
CMD_SLEEP = "sleep"
CMD_OPEN_APP = "open_app"
CMD_CLOSE_APP = "close_app"
CMD_MINIMIZE_APP = "minimize_app"
CMD_MAXIMIZE_APP = "maximize_app"
CMD_FOCUS_APP = "focus_app"
CMD_TYPE_TEXT = "type_text"
CMD_HOTKEY = "hotkey"
CMD_SCROLL = "scroll"

PC_COMMANDS = frozenset(
    {
        CMD_VOLUME_SET,
        CMD_VOLUME_UP,
        CMD_VOLUME_DOWN,
        CMD_MUTE,
        CMD_UNMUTE,
        CMD_MEDIA_PLAY_PAUSE,
        CMD_MEDIA_NEXT,
        CMD_MEDIA_PREV,
        CMD_DISPLAY_OFF,
        CMD_DISPLAY_ON,
        CMD_SLEEP,
        CMD_OPEN_APP,
        CMD_CLOSE_APP,
        CMD_MINIMIZE_APP,
        CMD_MAXIMIZE_APP,
        CMD_FOCUS_APP,
        CMD_TYPE_TEXT,
        CMD_HOTKEY,
        CMD_SCROLL,
    }
)

# --- mouse_click (SPEC §5 tool 6 / §8) ---------------------------------------
BUTTON_LEFT = "left"
BUTTON_RIGHT = "right"
BUTTON_DOUBLE = "double"

#: spoken/model variants accepted for the ``button`` argument
MOUSE_BUTTONS: dict[str, str] = {
    "": BUTTON_LEFT,
    "left": BUTTON_LEFT,
    "l": BUTTON_LEFT,
    "primary": BUTTON_LEFT,
    "click": BUTTON_LEFT,
    "right": BUTTON_RIGHT,
    "r": BUTTON_RIGHT,
    "secondary": BUTTON_RIGHT,
    "context": BUTTON_RIGHT,
    "double": BUTTON_DOUBLE,
    "double click": BUTTON_DOUBLE,
    "double_click": BUTTON_DOUBLE,
    "doubleclick": BUTTON_DOUBLE,
    "dblclick": BUTTON_DOUBLE,
    "left double": BUTTON_DOUBLE,
    "left_double": BUTTON_DOUBLE,
}

#: pause between the two clicks of a double click (below the Windows default
#: double-click time of 500 ms, far enough apart for slow UI frameworks)
DOUBLE_CLICK_GAP_S = 0.12

#: pause after moving the cursor so hover states settle before the button press
CLICK_SETTLE_S = 0.04

#: how long ``tasklist`` may take while looking up an app's process ids
TASKLIST_TIMEOUT_S = 15.0

#: how many window titles a ``minimize_app`` result names back to the LLM
MAX_REPORTED_WINDOWS = 3

#: volume step for ``volume_up`` / ``volume_down`` (5 %)
VOLUME_STEP = 0.05

#: how long to wait for the suspend thread to report a failure before answering
SLEEP_GRACE_S = 0.7

#: ``run_command`` limits (SPEC §5): wall clock and returned output size
RUN_COMMAND_TIMEOUT_S = 30.0
RUN_COMMAND_OUTPUT_LIMIT = 4000

#: upper bound for a single ``type_text`` action — a voice command never needs more
MAX_TYPE_CHARS = 4000

#: how many SendInput events are pushed in one call while typing
TYPE_CHUNK_EVENTS = 100

#: pause between typing chunks so slower windows keep up
TYPE_CHUNK_PAUSE_S = 0.005

# --- Win32 constants ---------------------------------------------------------
_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x01000
#: One wheel notch, as Windows counts it (WHEEL_DELTA).
_WHEEL_DELTA = 120

#: button name -> (press flag, release flag); "double" reuses the left pair
_MOUSE_BUTTON_EVENTS: dict[str, tuple[int, int]] = {
    BUTTON_LEFT: (_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
    BUTTON_RIGHT: (_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
    BUTTON_DOUBLE: (_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
}

#: GetSystemMetrics indices for the primary screen size
_SM_CXSCREEN = 0
_SM_CYSCREEN = 1

#: ShowWindow command and the window styles/attributes used by minimize_app
_SW_MINIMIZE = 6
_SW_RESTORE = 9
_SW_MAXIMIZE = 3
#: Hotkeys that close the focused window/tab. Refused when the focused window
#: is our own console — that is how Jarvis once closed itself instead of a tab.
_CLOSING_HOTKEYS = frozenset({"ctrl+w", "ctrl+f4", "ctrl+shift+w", "alt+f4", "ctrl+shift+q"})
_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080
#: DwmGetWindowAttribute index telling whether a window is cloaked — UWP apps
#: keep invisible "cloaked" frame windows around that must not count as open.
_DWMWA_CLOAKED = 14

VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_F1 = 0x70

_HWND_BROADCAST = 0xFFFF
_WM_SYSCOMMAND = 0x0112
_SC_MONITORPOWER = 0xF170
_MONITOR_POWER_OFF = 2

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

_IS_WINDOWS = sys.platform == "win32"


class PCActionError(RuntimeError):
    """Recoverable failure of a ``pc_control`` / ``run_command`` action.

    ``output`` carries extra data for the LLM (e.g. the closest app names when
    ``open_app`` could not resolve what the user said) — the dispatcher copies it
    into ``action_result.output`` (SPEC §4).
    """

    def __init__(self, message: str, output: str | None = None) -> None:
        super().__init__(message)
        self.output = output


class PCResult(NamedTuple):
    """Outcome of a successful command: log/summary text plus optional output."""

    detail: str
    output: str | None = None


# --- hotkey vocabulary (SPEC §8) ---------------------------------------------

#: modifier name -> virtual-key code
MODIFIER_KEYS: dict[str, int] = {
    "ctrl": VK_CONTROL,
    "control": VK_CONTROL,
    "alt": VK_MENU,
    "menu": VK_MENU,
    "shift": VK_SHIFT,
    "win": VK_LWIN,
    "windows": VK_LWIN,
    "super": VK_LWIN,
    "meta": VK_LWIN,
    "cmd": VK_LWIN,
}

#: named key -> (virtual-key code, extended-key flag)
NAMED_KEYS: dict[str, tuple[int, bool]] = {
    "enter": (VK_RETURN, False),
    "return": (VK_RETURN, False),
    "esc": (VK_ESCAPE, False),
    "escape": (VK_ESCAPE, False),
    "tab": (VK_TAB, False),
    "space": (VK_SPACE, False),
    "spacebar": (VK_SPACE, False),
    "backspace": (VK_BACK, False),
    "bksp": (VK_BACK, False),
    "del": (VK_DELETE, True),
    "delete": (VK_DELETE, True),
    "up": (VK_UP, True),
    "down": (VK_DOWN, True),
    "left": (VK_LEFT, True),
    "right": (VK_RIGHT, True),
}

_HOTKEY_HELP = (
    "supported: ctrl, alt, shift, win + a letter, a digit, f1-f24, enter, esc, "
    "tab, space, backspace, del, up, down, left, right"
)

# --- ctypes plumbing ---------------------------------------------------------

ULONG_PTR = wintypes.WPARAM


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT))


class _INPUT(ctypes.Structure):
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUTUNION))


#: EnumWindows callback signature: ``BOOL (HWND hwnd, LPARAM lparam)``
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class _WindowInfo(NamedTuple):
    """One visible top-level window found by :func:`_list_windows`."""

    hwnd: int
    pid: int
    title: str


_dll_lock = threading.Lock()
_user32_dll: Any = None
_dwmapi_dll: Any = None


def _require_windows() -> None:
    if not _IS_WINDOWS:
        raise PCActionError("pc_control commands only work on Windows")


def _user32() -> Any:
    """Return a configured ``user32`` handle (loaded once, thread-safe)."""

    global _user32_dll
    _require_windows()
    with _dll_lock:
        if _user32_dll is None:
            dll = ctypes.WinDLL("user32", use_last_error=True)
            dll.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
            dll.SendInput.restype = wintypes.UINT
            dll.SendMessageW.argtypes = (
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            dll.SendMessageW.restype = wintypes.LPARAM
            # window enumeration / minimizing (SPEC §8, minimize_app)
            dll.EnumWindows.argtypes = (_WNDENUMPROC, wintypes.LPARAM)
            dll.EnumWindows.restype = wintypes.BOOL
            dll.GetWindowThreadProcessId.argtypes = (
                wintypes.HWND,
                ctypes.POINTER(wintypes.DWORD),
            )
            dll.GetWindowThreadProcessId.restype = wintypes.DWORD
            dll.IsWindowVisible.argtypes = (wintypes.HWND,)
            dll.IsWindowVisible.restype = wintypes.BOOL
            dll.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
            dll.ShowWindow.restype = wintypes.BOOL
            dll.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
            dll.GetWindowTextLengthW.restype = ctypes.c_int
            dll.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
            dll.GetWindowTextW.restype = ctypes.c_int
            dll.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
            dll.GetWindowLongW.restype = wintypes.LONG
            # cursor / screen geometry (SPEC §8, mouse_click)
            dll.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
            dll.SetCursorPos.restype = wintypes.BOOL
            dll.GetSystemMetrics.argtypes = (ctypes.c_int,)
            dll.GetSystemMetrics.restype = ctypes.c_int
            _user32_dll = dll
        return _user32_dll


def _dwmapi() -> Any:
    """Return a configured ``dwmapi`` handle, or ``None`` when unavailable."""

    global _dwmapi_dll
    if not _IS_WINDOWS:
        return None
    with _dll_lock:
        if _dwmapi_dll is None:
            try:
                dll = ctypes.WinDLL("dwmapi")
            except OSError as exc:  # pragma: no cover - dwmapi ships with Vista+
                log.debug("dwmapi is unavailable, cloaked windows will count: %s", exc)
                _dwmapi_dll = False
                return None
            dll.DwmGetWindowAttribute.argtypes = (
                wintypes.HWND,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
            )
            dll.DwmGetWindowAttribute.restype = ctypes.c_long
            _dwmapi_dll = dll
        return _dwmapi_dll or None


def _send_input(*events: _INPUT) -> None:
    _send_input_batch(events)


def _send_input_batch(events: Sequence[_INPUT]) -> None:
    """Push a batch of input events in one ``SendInput`` call (order preserved)."""

    count = len(events)
    if not count:
        return
    user32 = _user32()
    array = (_INPUT * count)(*events)
    sent = user32.SendInput(count, array, ctypes.sizeof(_INPUT))
    if sent != count:
        raise PCActionError(
            f"SendInput delivered {sent} of {count} events "
            f"(error code {ctypes.get_last_error()})"
        )


def _key_event(vk: int, key_up: bool, extended: bool = False) -> _INPUT:
    flags = (_KEYEVENTF_EXTENDEDKEY if extended else 0) | (
        _KEYEVENTF_KEYUP if key_up else 0
    )
    return _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUTUNION(
            ki=_KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
        ),
    )


def _unicode_event(code_unit: int, key_up: bool) -> _INPUT:
    """A KEYEVENTF_UNICODE event carrying one UTF-16 code unit."""

    flags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if key_up else 0)
    return _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUTUNION(
            ki=_KEYBDINPUT(wVk=0, wScan=code_unit, dwFlags=flags, time=0, dwExtraInfo=0)
        ),
    )


def _mouse_move_event(dx: int, dy: int) -> _INPUT:
    return _INPUT(
        type=_INPUT_MOUSE,
        union=_INPUTUNION(
            mi=_MOUSEINPUT(
                dx=dx, dy=dy, mouseData=0, dwFlags=_MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0
            )
        ),
    )


def _mouse_wheel_event(notches: int, horizontal: bool = False) -> _INPUT:
    """One wheel event at the cursor: positive is up / right, negative is down / left."""

    return _INPUT(
        type=_INPUT_MOUSE,
        union=_INPUTUNION(
            mi=_MOUSEINPUT(
                dx=0,
                dy=0,
                # mouseData is a SIGNED amount, but the field is DWORD: hand
                # Windows the two's-complement value for a downward scroll.
                mouseData=(notches * _WHEEL_DELTA) & 0xFFFFFFFF,
                dwFlags=_MOUSEEVENTF_HWHEEL if horizontal else _MOUSEEVENTF_WHEEL,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def _mouse_button_event(flags: int) -> _INPUT:
    """A button press/release at the cursor's current position (no movement)."""

    return _INPUT(
        type=_INPUT_MOUSE,
        union=_INPUTUNION(
            mi=_MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
        ),
    )


@contextmanager
def _com_apartment() -> Iterator[None]:
    """Initialise COM for the current (worker) thread for the pycaw calls."""

    try:
        import comtypes  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PCActionError(
            "the comtypes library is missing (pip install comtypes pycaw)"
        ) from exc

    initialized = False
    try:
        comtypes.CoInitialize()
        initialized = True
    except OSError as exc:
        # RPC_E_CHANGED_MODE etc. — the thread already lives in another apartment
        log.debug("CoInitialize failed, continuing without it: %s", exc)
    try:
        yield
    finally:
        if initialized:
            try:
                comtypes.CoUninitialize()
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask errors
                log.debug("CoUninitialize failed: %s", exc)


def _endpoint_volume() -> Any:
    """Return ``IAudioEndpointVolume`` for the default output device.

    Recent pycaw releases hand back an ``AudioDevice`` wrapper with a ready
    ``EndpointVolume`` property, older ones a raw ``IMMDevice`` that still has to
    be activated — both are supported here.
    """

    try:
        from comtypes import CLSCTX_ALL  # type: ignore[import-not-found]
        from pycaw.pycaw import (  # type: ignore[import-not-found]
            AudioUtilities,
            IAudioEndpointVolume,
        )
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise PCActionError(
            "the pycaw library is missing (pip install pycaw comtypes)"
        ) from exc

    speakers = AudioUtilities.GetSpeakers()
    if speakers is None:
        raise PCActionError("no audio output device found")

    if hasattr(type(speakers), "EndpointVolume"):
        return speakers.EndpointVolume

    activate = getattr(speakers, "Activate", None)
    if not callable(activate):
        raise PCActionError(
            "pycaw returned an unexpected speaker object "
            f"({type(speakers).__name__}) — cannot reach the volume interface"
        )
    interface = activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))


# --- blocking workers (run via asyncio.to_thread) ----------------------------


def _clamp_scalar(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sync_set_volume(scalar: float) -> float:
    with _com_apartment():
        volume = _endpoint_volume()
        level = _clamp_scalar(scalar)
        volume.SetMasterVolumeLevelScalar(level, None)
        if level > 0.0:
            volume.SetMute(0, None)
        return level


def _sync_step_volume(delta: float) -> float:
    with _com_apartment():
        volume = _endpoint_volume()
        current = float(volume.GetMasterVolumeLevelScalar())
        level = _clamp_scalar(current + delta)
        volume.SetMasterVolumeLevelScalar(level, None)
        if delta > 0.0 and level > 0.0:
            volume.SetMute(0, None)
        return level


def _sync_set_mute(muted: bool) -> None:
    with _com_apartment():
        volume = _endpoint_volume()
        volume.SetMute(1 if muted else 0, None)


def _sync_media_key(vk: int) -> None:
    # media keys are extended keys on a PC/AT keyboard
    _send_input(_key_event(vk, key_up=False, extended=True), _key_event(vk, key_up=True, extended=True))


def _sync_display_off() -> None:
    user32 = _user32()
    user32.SendMessageW(
        _HWND_BROADCAST, _WM_SYSCOMMAND, _SC_MONITORPOWER, _MONITOR_POWER_OFF
    )


def _sync_display_on() -> None:
    # a 1 px mouse move is the reliable way to wake the monitor back up
    _send_input(_mouse_move_event(1, 0))
    time.sleep(0.05)
    _send_input(_mouse_move_event(-1, 0))


def _sync_suspend() -> None:
    _require_windows()
    powrprof = ctypes.WinDLL("powrprof", use_last_error=True)
    powrprof.SetSuspendState.argtypes = (ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte)
    powrprof.SetSuspendState.restype = ctypes.c_ubyte
    # hibernate=False, force=True, wakeup events enabled
    result = powrprof.SetSuspendState(0, 1, 0)
    if not result:
        raise PCActionError(
            f"SetSuspendState failed (error code {ctypes.get_last_error()})"
        )


def _text_events(text: str) -> list[_INPUT]:
    """Build the SendInput events that type ``text`` (SPEC §8).

    Every character is sent as UTF-16 code units with ``KEYEVENTF_UNICODE``, so
    surrogate pairs (emoji, rare CJK) become two consecutive code-unit events —
    Windows joins them into one character. Line breaks and tabs have no unicode
    equivalent that applications act on, so they are sent as real key presses.
    """

    events: list[_INPUT] = []
    for char in text:
        if char == "\r":
            continue
        if char == "\n":
            events.append(_key_event(VK_RETURN, key_up=False))
            events.append(_key_event(VK_RETURN, key_up=True))
            continue
        if char == "\t":
            events.append(_key_event(VK_TAB, key_up=False))
            events.append(_key_event(VK_TAB, key_up=True))
            continue
        units = _utf16_units(char)
        for unit in units:
            events.append(_unicode_event(unit, key_up=False))
        for unit in units:
            events.append(_unicode_event(unit, key_up=True))
    return events


def _utf16_units(char: str) -> tuple[int, ...]:
    """Return the UTF-16 code units of one character (1, or 2 for a surrogate pair)."""

    encoded = char.encode("utf-16-le")
    return tuple(
        encoded[index] | (encoded[index + 1] << 8) for index in range(0, len(encoded), 2)
    )


def _sync_type_text(text: str) -> int:
    """Type ``text`` into the focused window. Returns the character count."""

    _require_windows()
    events = _text_events(text)
    for start in range(0, len(events), TYPE_CHUNK_EVENTS):
        _send_input_batch(events[start : start + TYPE_CHUNK_EVENTS])
        if start + TYPE_CHUNK_EVENTS < len(events):
            time.sleep(TYPE_CHUNK_PAUSE_S)
    return len(text)


def parse_hotkey(combo: Any) -> tuple[list[int], list[tuple[int, bool]], str]:
    """Parse ``"ctrl+shift+t"`` into ``(modifier vks, [(vk, extended)], label)``.

    Raises :class:`PCActionError` for an empty combo or an unsupported key name.
    """

    text = "" if combo is None else str(combo)
    parts = [part.strip().casefold() for part in text.replace(" ", "+").split("+")]
    parts = [part for part in parts if part]
    if not parts:
        raise PCActionError("no hotkey given (value), e.g. 'ctrl+shift+t'")

    modifiers: list[int] = []
    keys: list[tuple[int, bool]] = []
    labels: list[str] = []
    for part in parts:
        labels.append(part)
        modifier = MODIFIER_KEYS.get(part)
        if modifier is not None:
            if modifier not in modifiers:
                modifiers.append(modifier)
            continue
        named = NAMED_KEYS.get(part)
        if named is not None:
            keys.append(named)
            continue
        if len(part) == 1 and (part.isalpha() or part.isdigit()) and part.isascii():
            keys.append((ord(part.upper()), False))
            continue
        if part.startswith("f") and part[1:].isdigit():
            number = int(part[1:])
            if 1 <= number <= 24:
                keys.append((VK_F1 + number - 1, False))
                continue
        raise PCActionError(f"unknown hotkey key '{part}' ({_HOTKEY_HELP})")

    if not modifiers and not keys:
        raise PCActionError(f"hotkey '{text}' has no keys to press ({_HOTKEY_HELP})")
    return modifiers, keys, "+".join(labels)


#: How many wheel notches "scroll down" means when no amount is given. Three is
#: what a physical wheel flick does and roughly a third of a page in a browser.
DEFAULT_SCROLL_NOTCHES = 3
#: Ceiling per call, so a model that asks to scroll "1000" cannot send the page
#: into orbit; it can always call scroll again.
MAX_SCROLL_NOTCHES = 30

_SCROLL_DOWN_WORDS = frozenset({"down", "d", "under", "below", "вниз", "ниже"})
_SCROLL_UP_WORDS = frozenset({"up", "u", "above", "back", "вверх", "выше"})
_SCROLL_HELP = "say for example 'down', 'up', 'down 5'"


def parse_scroll(value: Any) -> tuple[int, str]:
    """Read ``value`` into signed wheel notches plus a label for the reply.

    Accepts a direction (``"down"``, ``"up"``), a direction with an amount
    (``"down 5"``), or a bare signed number where negative means down. Pure
    parsing so it can be unit-tested without a desktop.

    :raises PCActionError: nothing usable in ``value``.
    """
    text = " ".join(str(value or "").split()).lower()
    if not text:
        return -DEFAULT_SCROLL_NOTCHES, f"down {DEFAULT_SCROLL_NOTCHES}"

    direction = 0
    amount: int | None = None
    for part in text.replace(",", " ").split():
        if part in _SCROLL_DOWN_WORDS:
            direction = -1
            continue
        if part in _SCROLL_UP_WORDS:
            direction = 1
            continue
        try:
            number = int(float(part))
        except ValueError:
            continue
        amount = abs(number)
        if direction == 0 and number < 0:
            direction = -1
        elif direction == 0 and number > 0:
            # A bare positive number after no direction word means "down N":
            # asking to "scroll 5" on a page always means further down it.
            direction = -1

    if direction == 0 and amount is None:
        raise PCActionError(f"unclear scroll {value!r} ({_SCROLL_HELP})")
    if direction == 0:
        direction = -1
    if amount is None or amount <= 0:
        amount = DEFAULT_SCROLL_NOTCHES
    amount = min(amount, MAX_SCROLL_NOTCHES)
    return direction * amount, f"{'up' if direction > 0 else 'down'} {amount}"


def _sync_scroll(notches: int) -> None:
    """Send ``notches`` wheel events, one at a time.

    One event per notch rather than a single big one: browsers and Explorer
    animate smooth scrolling per event, and a single 30-notch event either
    jumps the whole way at once or gets clamped.
    """
    step = 1 if notches > 0 else -1
    for _ in range(abs(notches)):
        _send_input(_mouse_wheel_event(step))
        time.sleep(0.01)


def _sync_hotkey(modifiers: Sequence[int], keys: Sequence[tuple[int, bool]]) -> None:
    """Press modifiers, then the keys, and release everything in reverse order."""

    _require_windows()
    # modifiers down -> keys down -> keys up -> modifiers up; a modifier-only
    # combo (e.g. "win") simply has no keys in the middle.
    events: list[_INPUT] = [_key_event(vk, key_up=False) for vk in modifiers]
    events.extend(_key_event(vk, key_up=False, extended=extended) for vk, extended in keys)
    events.extend(
        _key_event(vk, key_up=True, extended=extended) for vk, extended in reversed(keys)
    )
    events.extend(_key_event(vk, key_up=True) for vk in reversed(modifiers))
    _send_input_batch(events)


def truncate_output(text: str, limit: int = RUN_COMMAND_OUTPUT_LIMIT) -> str:
    """Cut command output down to ``limit`` characters, marking what was dropped."""

    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    marker = f"\n... [truncated, {len(clean)} chars total]"
    keep = max(0, limit - len(marker))
    return clean[:keep] + marker


def _kill_process_tree(pid: int) -> None:
    """Terminate a spawned process and everything it started (SPEC §8)."""

    try:
        subprocess.run(  # noqa: S603 - fixed command, pid from our own Popen
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=_CREATE_NO_WINDOW,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - best effort, the caller kills anyway
        log.warning("taskkill could not stop the process tree of pid %s: %s", pid, exc)


def _sync_run_command(command: str, timeout_s: float) -> tuple[int | None, str, bool]:
    """Run PowerShell. Returns ``(exit code, merged output, timed out)``."""

    _require_windows()
    argv = [
        powershell_executable(),
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        command,
    ]
    try:
        process = subprocess.Popen(  # noqa: S603 - the LLM's command, by design (SPEC §5)
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=_CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        raise PCActionError(f"could not start PowerShell: {exc}") from exc

    try:
        stdout, _ = process.communicate(timeout=timeout_s)
        return process.returncode, decode_console_output(stdout), False
    except subprocess.TimeoutExpired:
        _kill_process_tree(process.pid)
        try:
            stdout, _ = process.communicate(timeout=5)
        except Exception:  # noqa: BLE001 - the tree is already being killed
            stdout = b""
            try:
                process.kill()
            except Exception as exc:  # noqa: BLE001 - nothing left to do
                log.debug("could not kill pid %s: %s", process.pid, exc)
        return None, decode_console_output(stdout), True


def _sync_close_process(image_name: str) -> str:
    """``taskkill /IM <image> /F`` — returns the tool's output on success."""

    _require_windows()
    completed = subprocess.run(  # noqa: S603 - fixed command, name from the app index
        ["taskkill", "/IM", image_name, "/F"],
        capture_output=True,
        creationflags=_CREATE_NO_WINDOW,
        check=False,
    )
    output = (
        decode_console_output(completed.stdout).strip()
        or decode_console_output(completed.stderr).strip()
    )
    if completed.returncode != 0:
        detail = output or f"taskkill exited with code {completed.returncode}"
        raise PCActionError(f"could not close '{image_name}': {detail}")
    return output or f"process {image_name} terminated"


# --- windows: minimize_app (SPEC §8, v1.2) -----------------------------------


def _window_title(hwnd: Any) -> str:
    """Read a window's caption; empty string when it has none."""

    user32 = _user32()
    length = int(user32.GetWindowTextLengthW(hwnd))
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    if int(user32.GetWindowTextW(hwnd, buffer, length + 1)) <= 0:
        return ""
    return buffer.value


def _is_cloaked(hwnd: Any) -> bool:
    """True for a DWM-cloaked window (a suspended/background UWP frame).

    Those windows report ``IsWindowVisible() == TRUE`` while being nowhere on
    screen, so minimizing one would look like success without the user seeing
    anything happen.
    """

    dwmapi = _dwmapi()
    if dwmapi is None:
        return False
    cloaked = wintypes.DWORD(0)
    result = dwmapi.DwmGetWindowAttribute(
        hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
    )
    return result == 0 and cloaked.value != 0


def _list_windows() -> list[_WindowInfo]:
    """Every visible top-level window that belongs to a real application.

    ``EnumWindows`` walks the top-level windows of the desktop; tool windows,
    cloaked frames and captionless helper windows are dropped because none of
    them is something a user would think of as "the app's window".
    """

    _require_windows()
    user32 = _user32()
    windows: list[_WindowInfo] = []

    @_WNDENUMPROC
    def collect(hwnd: Any, _lparam: Any) -> bool:
        # An exception inside a ctypes callback cannot reach the caller, so a
        # window that misbehaves is skipped instead of breaking enumeration.
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            if int(user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)) & _WS_EX_TOOLWINDOW:
                return True
            title = _window_title(hwnd)
            if not title or _is_cloaked(hwnd):
                return True
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            windows.append(_WindowInfo(hwnd=int(hwnd), pid=int(pid.value), title=title))
        except Exception as exc:  # noqa: BLE001 - one bad window must not stop the walk
            log.debug("skipping a window during enumeration: %s", exc)
        return True

    ctypes.set_last_error(0)
    if not user32.EnumWindows(collect, 0) and not windows:
        code = ctypes.get_last_error()
        if code:
            raise PCActionError(f"EnumWindows failed (error code {code})")
    return windows


def _sync_process_ids(image_name: str) -> set[int]:
    """Process ids of every running instance of ``image_name`` (via ``tasklist``)."""

    _require_windows()
    try:
        completed = subprocess.run(  # noqa: S603 - fixed command, name from the app index
            ["tasklist", "/FO", "CSV", "/NH", "/FI", f"IMAGENAME eq {image_name}"],
            capture_output=True,
            creationflags=_CREATE_NO_WINDOW,
            timeout=TASKLIST_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("tasklist timed out while looking for '%s'", image_name)
        return set()
    except OSError as exc:
        log.warning("could not run tasklist for '%s': %s", image_name, exc)
        return set()

    if completed.returncode != 0:
        log.debug("tasklist exited with code %s for '%s'", completed.returncode, image_name)
        return set()

    # CSV rows look like: "chrome.exe","12345","Console","1","250 000 K".
    # When nothing matches, tasklist prints an INFO line instead of rows.
    wanted = image_name.strip().casefold()
    pids: set[int] = set()
    for row in csv.reader(io.StringIO(decode_console_output(completed.stdout))):
        if len(row) < 2 or row[0].strip().casefold() != wanted:
            continue
        digits = "".join(char for char in row[1] if char.isdigit())
        if digits:
            pids.add(int(digits))
    return pids


#: vendor words that say nothing about which app a window belongs to
_GENERIC_NAME_WORDS = frozenset(
    {"microsoft", "windows", "google", "app", "the", "and", "for", "new"}
)

#: punctuation trimmed off a word before comparing captions to app names
_WORD_TRIM = ".,:;!?()[]{}\"'|*<>/\\–—"


def _name_words(text: str) -> set[str]:
    """Significant words of a normalised caption/app name (4+ chars, no vendor)."""

    words = set()
    for raw in text.split():
        word = raw.strip(_WORD_TRIM)
        if len(word) >= 4 and word not in _GENERIC_NAME_WORDS:
            words.add(word)
    return words


def _title_matches(title: str, query: str) -> bool:
    """Match a window caption against a normalised app name (UWP fallback)."""

    normalized = normalize_app_name(title)
    if not normalized or not query:
        return False
    if query in normalized:
        return True
    # "Photos" (window) vs "Microsoft Photos" (Start Menu name); short captions
    # like "a" would match anything, so they do not count.
    if len(normalized) >= 3 and normalized in query:
        return True
    # Store apps usually put the document first and drop the vendor prefix
    # ("image.png - Photos"), so one shared significant word is enough.
    return bool(_name_words(normalized) & _name_words(query))


#: Spoken names for "the console/terminal", and the exes that host one. The
#: Jarvis client itself runs in one of these, so "hide the console" must reach
#: it even though no such Start-menu app exists.
_CONSOLE_ALIASES = frozenset(
    {
        "console",
        "the console",
        "terminal",
        "the terminal",
        "command prompt",
        "the command prompt",
        "cmd",
        "cmd prompt",
        "powershell",
        "power shell",
        "windows terminal",
        "shell",
    }
)
_CONSOLE_EXES = ("cmd.exe", "powershell.exe", "pwsh.exe", "windowsterminal.exe", "conhost.exe")


def _is_console_alias(value: Any) -> bool:
    return normalize_app_name(str(value or "")) in _CONSOLE_ALIASES


def _sync_minimize_console() -> list[str]:
    """Minimize every visible console/terminal window (SPEC §8, minimize_app)."""

    _require_windows()
    pids: set[int] = set()
    for exe in _CONSOLE_EXES:
        pids |= _sync_process_ids(exe)
    if not pids:
        return []
    user32 = _user32()
    minimized: list[str] = []
    for window in _list_windows():
        if window.pid in pids:
            user32.ShowWindow(window.hwnd, _SW_MINIMIZE)
            minimized.append(window.title)
    return minimized


def _sync_focus_window(hwnd: int) -> None:
    """Restore a window and bring it to the foreground.

    Windows refuses ``SetForegroundWindow`` from a background process unless an
    Alt key event "unlocks" it first — the long-standing ``keybd_event``
    workaround, needed because the client runs headless.
    """
    user32 = _user32()
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.keybd_event.argtypes = (
        ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_void_p,
    )
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    # SW_RESTORE would also un-maximize a maximized window (it shrank the
    # browser once) - only restore when the window is actually minimized.
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, _SW_RESTORE)
    user32.keybd_event(0x12, 0, 0, None)       # VK_MENU down
    user32.keybd_event(0x12, 0, 0x0002, None)  # VK_MENU up
    user32.SetForegroundWindow(hwnd)


def _sync_focus_app(image_name: str | None, display_name: str) -> str | None:
    """Bring the app's main window to the foreground; returns its title."""
    _require_windows()
    pids = _sync_process_ids(image_name) if image_name else set()
    windows = _list_windows()
    targets = [window for window in windows if window.pid in pids] if pids else []
    if not targets:
        query = normalize_app_name(display_name)
        targets = [window for window in windows if _title_matches(window.title, query)]
    if not targets:
        return None
    _sync_focus_window(targets[0].hwnd)
    return targets[0].title


def _sync_maximize_app(image_name: str | None, display_name: str) -> str | None:
    """Maximize the app's main window and bring it to the foreground."""
    _require_windows()
    pids = _sync_process_ids(image_name) if image_name else set()
    windows = _list_windows()
    targets = [window for window in windows if window.pid in pids] if pids else []
    if not targets:
        query = normalize_app_name(display_name)
        targets = [window for window in windows if _title_matches(window.title, query)]
    if not targets:
        return None
    user32 = _user32()
    user32.ShowWindow(targets[0].hwnd, _SW_MAXIMIZE)
    _sync_focus_window(targets[0].hwnd)
    return targets[0].title


def _own_console_focused() -> bool:
    """True when the foreground window is the console hosting this client."""
    try:
        user32 = _user32()
        kernel32 = ctypes.windll.kernel32
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        user32.GetForegroundWindow.restype = wintypes.HWND
        own = kernel32.GetConsoleWindow()
        return bool(own) and user32.GetForegroundWindow() == own
    except Exception as exc:  # noqa: BLE001 - a guard must never break the action
        log.debug("could not compare console windows: %s", exc)
        return False


def _sync_minimize_app(image_name: str | None, display_name: str) -> list[str]:
    """Minimize every visible window of an app. Returns the titles minimized.

    Desktop apps are found by process id (``tasklist`` on the exe name that the
    app index resolved); Store/UWP entries have no exe, so their windows are
    matched by caption against the app's display name (SPEC §8).
    """

    _require_windows()
    pids = _sync_process_ids(image_name) if image_name else set()
    windows = _list_windows()

    targets = [window for window in windows if window.pid in pids] if pids else []
    if not targets:
        query = normalize_app_name(display_name)
        targets = [window for window in windows if _title_matches(window.title, query)]

    user32 = _user32()
    minimized: list[str] = []
    for window in targets:
        user32.ShowWindow(window.hwnd, _SW_MINIMIZE)
        minimized.append(window.title)
    return minimized


# --- mouse_click (SPEC §5 tool 6 / §8, v1.2) ---------------------------------


def _sync_mouse_click(x_norm: float, y_norm: float, button: str) -> tuple[int, int, int, int]:
    """Move the cursor to a normalized point and click. Returns ``(x, y, w, h)``."""

    _require_windows()
    user32 = _user32()
    screen_w = int(user32.GetSystemMetrics(_SM_CXSCREEN))
    screen_h = int(user32.GetSystemMetrics(_SM_CYSCREEN))
    if screen_w <= 0 or screen_h <= 0:
        raise PCActionError("could not read the screen resolution (GetSystemMetrics returned 0)")

    # 1.0 maps to the last pixel, not one past the right/bottom edge.
    x_px = max(0, min(screen_w - 1, int(x_norm * screen_w)))
    y_px = max(0, min(screen_h - 1, int(y_norm * screen_h)))
    if not user32.SetCursorPos(x_px, y_px):
        raise PCActionError(
            f"SetCursorPos({x_px}, {y_px}) failed (error code {ctypes.get_last_error()})"
        )
    time.sleep(CLICK_SETTLE_S)

    down, up = _MOUSE_BUTTON_EVENTS[button]
    clicks = 2 if button == BUTTON_DOUBLE else 1
    for index in range(clicks):
        if index:
            time.sleep(DOUBLE_CLICK_GAP_S)
        _send_input(_mouse_button_event(down), _mouse_button_event(up))
    return x_px, y_px, screen_w, screen_h


def parse_click_coordinate(value: Any, field: str) -> float:
    """Parse a normalized 0..1 screen coordinate coming from the server."""

    if value is None or isinstance(value, bool):
        raise PCActionError(f"mouse_click needs '{field}' as a number between 0 and 1")
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        try:
            number = float(text)
        except ValueError as exc:
            raise PCActionError(
                f"unclear mouse_click {field} {value!r}: expected a number between 0 and 1"
            ) from exc
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise PCActionError(
            f"unclear mouse_click {field} {value!r}: expected a number between 0 and 1"
        )
    if not 0.0 <= number <= 1.0:
        raise PCActionError(
            f"mouse_click {field}={number} is outside 0..1 "
            f"(coordinates are fractions of the screen, not pixels)"
        )
    return number


def parse_mouse_button(value: Any) -> str:
    """Normalise the ``button`` argument to left / right / double."""

    text = "" if value is None else str(value).strip().casefold().replace("-", " ")
    button = MOUSE_BUTTONS.get(text)
    if button is None:
        raise PCActionError(
            f"unknown mouse button '{value}' (supported: "
            f"{BUTTON_LEFT}, {BUTTON_RIGHT}, {BUTTON_DOUBLE})"
        )
    return button


# --- controller --------------------------------------------------------------


def _parse_volume_value(value: Any) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PCActionError("volume_set needs a level between 0 and 100")
    if isinstance(value, bool):
        raise PCActionError(f"unclear volume level {value!r}: expected 0..100")
    if isinstance(value, str):
        text = value.strip().rstrip("%").replace(",", ".").strip()
        try:
            number = float(text)
        except ValueError as exc:
            raise PCActionError(f"unclear volume level {value!r}: expected 0..100") from exc
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise PCActionError(f"unclear volume level {value!r}: expected 0..100")
    if 0.0 < number <= 1.0 and isinstance(value, float):
        # tolerate a 0..1 scalar coming from the model
        number *= 100.0
    return max(0, min(100, int(round(number))))


class PCController:
    """Executes ``pc_control`` commands and ``run_command`` on the client machine."""

    def __init__(
        self,
        apps: Mapping[str, Any] | None = None,
        app_index: AppIndex | None = None,
    ) -> None:
        self.apps = app_index if app_index is not None else AppIndex(apps)

    # -- lifecycle ----------------------------------------------------------

    async def prepare(self) -> None:
        """Build the installed-app index ahead of the first ``open_app``."""

        await self.apps.ensure_ready()

    # -- pc_control ---------------------------------------------------------

    async def execute(self, command: Any, value: Any = None) -> PCResult:
        """Run one ``pc_control`` command. Raises :class:`PCActionError` on failure."""

        name = command.strip().casefold() if isinstance(command, str) else ""
        if not name:
            raise PCActionError("no pc_control command given")
        if name not in PC_COMMANDS:
            known = ", ".join(sorted(PC_COMMANDS))
            raise PCActionError(f"unknown pc_control command '{command}' (known: {known})")
        _require_windows()

        if name == CMD_VOLUME_SET:
            level = _parse_volume_value(value)
            await asyncio.to_thread(_sync_set_volume, level / 100.0)
            return PCResult(f"volume {level}%")

        if name in (CMD_VOLUME_UP, CMD_VOLUME_DOWN):
            delta = VOLUME_STEP if name == CMD_VOLUME_UP else -VOLUME_STEP
            level = await asyncio.to_thread(_sync_step_volume, delta)
            return PCResult(f"volume {int(round(level * 100))}%")

        if name in (CMD_MUTE, CMD_UNMUTE):
            muted = name == CMD_MUTE
            await asyncio.to_thread(_sync_set_mute, muted)
            return PCResult("sound muted" if muted else "sound unmuted")

        if name in (CMD_MEDIA_PLAY_PAUSE, CMD_MEDIA_NEXT, CMD_MEDIA_PREV):
            vk = {
                CMD_MEDIA_PLAY_PAUSE: VK_MEDIA_PLAY_PAUSE,
                CMD_MEDIA_NEXT: VK_MEDIA_NEXT_TRACK,
                CMD_MEDIA_PREV: VK_MEDIA_PREV_TRACK,
            }[name]
            await asyncio.to_thread(_sync_media_key, vk)
            return PCResult(f"media key {name}")

        if name == CMD_DISPLAY_OFF:
            await asyncio.to_thread(_sync_display_off)
            return PCResult("display off")

        if name == CMD_DISPLAY_ON:
            await asyncio.to_thread(_sync_display_on)
            return PCResult("display on")

        if name == CMD_SLEEP:
            return PCResult(await self._suspend())

        if name == CMD_TYPE_TEXT:
            return await self._type_text(value)

        if name == CMD_HOTKEY:
            return await self._hotkey(value)

        if name == CMD_SCROLL:
            return await self._scroll(value)

        if name == CMD_OPEN_APP:
            return await self._open_app(value)

        if name == CMD_MINIMIZE_APP:
            return await self.minimize_app(value)

        if name == CMD_FOCUS_APP:
            return await self.focus_app(value)

        if name == CMD_MAXIMIZE_APP:
            return await self.maximize_app(value)

        # CMD_CLOSE_APP
        return await self._close_app(value)

    # -- run_command --------------------------------------------------------

    async def run_command(self, command: Any) -> tuple[bool, str | None, str | None]:
        """Run a PowerShell command (SPEC §5). Returns ``(ok, error, output)``."""

        text = "" if command is None else str(command).strip()
        if not text:
            raise PCActionError("run_command needs a 'command' string")
        _require_windows()

        log.info("run_command: %s", text)
        code, output, timed_out = await asyncio.to_thread(
            _sync_run_command, text, RUN_COMMAND_TIMEOUT_S
        )
        clipped = truncate_output(output) or None

        if timed_out:
            message = f"command timed out after {int(RUN_COMMAND_TIMEOUT_S)} s and was terminated"
            log.warning("run_command: %s", message)
            return False, message, clipped
        if code != 0:
            message = f"command exited with code {code}"
            log.warning("run_command: %s", message)
            return False, message, clipped
        log.info("run_command: finished, %d chars of output", len(clipped or ""))
        return True, None, clipped

    # -- minimize_app / mouse_click (v1.2) ----------------------------------

    async def minimize_app(self, value: Any) -> PCResult:
        """Minimize every window of an app (``pc_control`` ``minimize_app``)."""

        # A console/terminal is not a Start-menu app: "hide the console" must
        # reach whatever shell hosts the Jarvis logs (cmd, PowerShell, Windows
        # Terminal), which the app index cannot resolve.
        if _is_console_alias(value):
            titles = await asyncio.to_thread(_sync_minimize_console)
            if not titles:
                raise PCActionError("no open console or terminal window found")
            log.info("pc_control: minimized %d console/terminal window(s)", len(titles))
            named = ", ".join(titles[:MAX_REPORTED_WINDOWS])
            return PCResult(f"minimized {len(titles)} console window(s): {named}")

        entry = await self._resolve_window_app(value)
        titles = await asyncio.to_thread(
            _sync_minimize_app, entry.process_name(), entry.name
        )
        if not titles:
            raise PCActionError(
                f"no open window found for '{entry.name}' - it does not seem to be running"
            )
        log.info("pc_control: minimized %d window(s) of '%s'", len(titles), entry.name)
        named = ", ".join(titles[:MAX_REPORTED_WINDOWS])
        detail = f"minimized {len(titles)} window(s) of {entry.name}"
        return PCResult(f"{detail}: {named}" if named else detail)

    async def focus_app(self, value: Any) -> PCResult:
        """Bring an app's window to the foreground (``pc_control`` ``focus_app``)."""

        if _is_console_alias(value):
            raise PCActionError(
                "refusing to focus the console - it hosts the Jarvis client itself"
            )
        entry = await self._resolve_window_app(value)
        title = await asyncio.to_thread(
            _sync_focus_app, entry.process_name(), entry.name
        )
        if title is None:
            raise PCActionError(
                f"no open window found for '{entry.name}' - open it first with open_app"
            )
        log.info("pc_control: focused '%s' (%s)", entry.name, title)
        return PCResult(f"focused {entry.name}: {title}")

    async def maximize_app(self, value: Any) -> PCResult:
        """Maximize an app's window (``pc_control`` ``maximize_app``)."""

        if _is_console_alias(value):
            raise PCActionError("refusing to maximize the console hosting the client")
        entry = await self._resolve_window_app(value)
        title = await asyncio.to_thread(
            _sync_maximize_app, entry.process_name(), entry.name
        )
        if title is None:
            raise PCActionError(
                f"no open window found for '{entry.name}' - open it first with open_app"
            )
        log.info("pc_control: maximized '%s' (%s)", entry.name, title)
        return PCResult(f"maximized {entry.name}: {title}")

    async def mouse_click(
        self, x_norm: Any, y_norm: Any, button: Any = None
    ) -> PCResult:
        """Click a normalized screen point (the server's ``click_screen`` tool).

        ``x_norm``/``y_norm`` are fractions of the screen (0..1) computed by the
        server from the vision model's answer, so this side owns the only real
        pixel arithmetic: the desktop resolution read from ``GetSystemMetrics``.
        """

        x_value = parse_click_coordinate(x_norm, "x_norm")
        y_value = parse_click_coordinate(y_norm, "y_norm")
        name = parse_mouse_button(button)
        _require_windows()

        x_px, y_px, screen_w, screen_h = await asyncio.to_thread(
            _sync_mouse_click, x_value, y_value, name
        )
        log.info(
            "mouse_click: %s click at %d,%d of %dx%d (%.3f, %.3f)",
            name, x_px, y_px, screen_w, screen_h, x_value, y_value,
        )
        return PCResult(f"{name} click at {x_px},{y_px} on a {screen_w}x{screen_h} screen")

    # -- helpers ------------------------------------------------------------

    async def _suspend(self) -> str:
        """Start the suspend in a background thread.

        ``SetSuspendState`` only returns once the machine resumes, so waiting for
        it would hang the action; instead we give the thread a short grace period
        to report an immediate failure.
        """

        failure: dict[str, str] = {}

        def worker() -> None:
            try:
                _sync_suspend()
            except Exception as exc:  # noqa: BLE001 - reported through the box/log
                failure["error"] = str(exc)
                log.error("pc_control: could not suspend the PC: %s", exc)

        thread = threading.Thread(target=worker, name="jarvis-suspend", daemon=True)
        thread.start()
        await asyncio.sleep(SLEEP_GRACE_S)
        if "error" in failure:
            raise PCActionError(failure["error"])
        return "the PC is going to sleep"

    async def _type_text(self, value: Any) -> PCResult:
        text = "" if value is None else str(value)
        if not text:
            raise PCActionError("type_text needs the text to type in 'value'")
        if len(text) > MAX_TYPE_CHARS:
            raise PCActionError(
                f"type_text got {len(text)} characters, the limit is {MAX_TYPE_CHARS}"
            )
        typed = await asyncio.to_thread(_sync_type_text, text)
        log.info("pc_control: typed %d characters", typed)
        return PCResult(f"typed {typed} characters")

    async def _hotkey(self, value: Any) -> PCResult:
        modifiers, keys, label = parse_hotkey(value)
        combo = "+".join(part.strip().lower() for part in str(value or "").split("+"))
        if combo in _CLOSING_HOTKEYS and await asyncio.to_thread(_own_console_focused):
            raise PCActionError(
                f"refusing to press {label}: the focused window is my own console "
                "and that key combination would close me - focus_app the target "
                "application first, then retry"
            )
        await asyncio.to_thread(_sync_hotkey, modifiers, keys)
        log.info("pc_control: pressed %s", label)
        return PCResult(f"pressed {label}")

    async def _scroll(self, value: Any) -> PCResult:
        """Turn the mouse wheel over whatever window is under the cursor."""
        notches, label = parse_scroll(value)
        await asyncio.to_thread(_sync_scroll, notches)
        log.info("pc_control: scrolled %s", label)
        return PCResult(f"scrolled {label}")

    async def _resolve_window_app(self, value: Any) -> AppEntry:
        """Window operations resolve the running browser, not Start-menu aliases."""
        from .app_control import GENERIC_BROWSER, BROWSERS, visible_apps
        from .apps import SOURCE_CONFIG
        if str(value or '').strip().casefold() not in GENERIC_BROWSER:
            return await self._resolve_app(value)
        browsers = [row for row in await asyncio.to_thread(visible_apps) if row['image'] in BROWSERS]
        if not browsers:
            raise PCActionError('No browser window is open. Ask which browser to open first.')
        if len(browsers) > 1:
            choices = ', '.join(row['name'] for row in browsers)
            raise PCActionError(f'Multiple browsers are open: {choices}. Ask which browser to use.')
        # Only used for window operations. Never launch a process from this row.
        return AppEntry(name=browsers[0]['name'], target=browsers[0]['image'], source=SOURCE_CONFIG)

    async def _resolve_app(self, value: Any) -> AppEntry:
        """Resolve an app name through the index, or raise with close matches."""

        if value is None or not str(value).strip():
            raise PCActionError("no application name given (value)")
        name = str(value).strip()
        entry = await self.apps.resolve(name)
        if entry is None:
            hint = self.apps.miss_hint(name)
            raise PCActionError(
                f"no installed application matches '{name}' ({hint})", output=hint
            )
        return entry

    async def _open_app(self, value: Any) -> PCResult:
        entry = await self._resolve_app(value)
        try:
            launched = await asyncio.to_thread(entry.launch)
        except AppError as exc:
            raise PCActionError(str(exc)) from exc
        log.info("pc_control: launched '%s' (%s)", entry.name, launched)
        return PCResult(f"opened: {entry.name}")

    async def _close_app(self, value: Any) -> PCResult:
        entry = await self._resolve_app(value)
        image_name = entry.process_name()
        if image_name is None:
            kind = "a Store app" if entry.is_store_app else "registered by AppID only"
            message = (
                f"'{entry.name}' is {kind} ('{entry.target}'), so close_app cannot map "
                f"it to a process image; use run_command with something like "
                f"\"Stop-Process -Name <process> -Force\" instead"
            )
            raise PCActionError(message, output=message)
        detail = await asyncio.to_thread(_sync_close_process, image_name)
        log.info("pc_control: closed '%s' (%s)", entry.name, image_name)
        return PCResult(f"closed: {entry.name} ({detail})")


__all__ = [
    "BUTTON_DOUBLE",
    "BUTTON_LEFT",
    "BUTTON_RIGHT",
    "MODIFIER_KEYS",
    "MOUSE_BUTTONS",
    "NAMED_KEYS",
    "PC_COMMANDS",
    "PCActionError",
    "PCController",
    "PCResult",
    "RUN_COMMAND_OUTPUT_LIMIT",
    "RUN_COMMAND_TIMEOUT_S",
    "parse_click_coordinate",
    "parse_hotkey",
    "parse_mouse_button",
    "truncate_output",
]
