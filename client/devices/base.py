"""Base abstractions for client-side smart-home devices (SPEC §8).

Every concrete driver (Magic Home, Tuya, SwitchBot Bot) subclasses :class:`Device`
and implements the methods that make sense for it; the rest keep the base
implementation, which raises ``NotImplementedError`` with a human readable
message that the dispatcher turns into an action error string.

The helpers in this module normalise the loose ``args`` produced by the LLM
(``state``/``brightness``/``color``/``action`` per SPEC §5) into strict Python
values, raising :class:`DeviceError` when the value cannot be understood.
"""

from __future__ import annotations

import logging
import re
from abc import ABC
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)

# --- light state / switch action vocabulary (values come from the tool schemas, SPEC §5)
STATE_ON = "on"
STATE_OFF = "off"
LIGHT_STATES = frozenset({STATE_ON, STATE_OFF})

SWITCH_ON = "on"
SWITCH_OFF = "off"
SWITCH_PRESS = "press"
SWITCH_TOGGLE = "toggle"
SWITCH_ACTIONS = frozenset({SWITCH_ON, SWITCH_OFF, SWITCH_PRESS, SWITCH_TOGGLE})

BRIGHTNESS_MIN = 1
BRIGHTNESS_MAX = 100


class DeviceError(RuntimeError):
    """Recoverable device-level failure.

    The dispatcher converts it into ``(False, "<message>", None)`` — it never
    escapes to the client main loop.
    """


# --- value parsing -----------------------------------------------------------

_ON_WORDS = {
    "on",
    "true",
    "1",
    "yes",
    "enable",
    "enabled",
    "turn on",
}
_OFF_WORDS = {
    "off",
    "false",
    "0",
    "no",
    "disable",
    "disabled",
    "turn off",
}

_PRESS_WORDS = {"press", "push", "click", "tap"}
_TOGGLE_WORDS = {"toggle", "switch", "flip"}

_HEX_RE = re.compile(r"^#?([0-9a-f]{3}|[0-9a-f]{6})$")
_RGB_CALL_RE = re.compile(
    r"^rgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$"
)

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "warm white": (255, 214, 170),
    "warm": (255, 214, 170),
    "cold white": (201, 226, 255),
    "cool white": (201, 226, 255),
    "cold": (201, 226, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
    "turquoise": (0, 255, 200),
    "yellow": (255, 220, 0),
    "orange": (255, 110, 0),
    "purple": (150, 0, 255),
    "violet": (150, 0, 255),
    "lilac": (180, 120, 255),
    "pink": (255, 60, 150),
    "magenta": (255, 0, 255),
    "crimson": (255, 0, 120),
    "lime": (160, 255, 0),
}


def normalize_name(value: Any) -> str:
    """Normalise a user/LLM supplied name for lookups (case, spacing, separators)."""

    text = "" if value is None else str(value)
    text = re.sub(r"[\s_\-]+", " ", text).strip().casefold()
    return text


def parse_state(value: Any, default: str | None = None) -> str:
    """Return ``"on"``/``"off"`` for a ``set_light`` ``state`` argument."""

    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return default
        raise DeviceError("no state given: expected 'on' or 'off'")
    if isinstance(value, bool):
        return STATE_ON if value else STATE_OFF
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return STATE_ON if float(value) > 0 else STATE_OFF
    text = normalize_name(value)
    if text in _ON_WORDS:
        return STATE_ON
    if text in _OFF_WORDS:
        return STATE_OFF
    raise DeviceError(f"unclear state={value!r}: expected 'on' or 'off'")


def parse_switch_action(value: Any, default: str | None = None) -> str:
    """Return one of ``on|off|press|toggle`` for a ``set_switch`` argument."""

    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return default
        raise DeviceError(
            "no action given: expected 'on', 'off', 'press' or 'toggle'"
        )
    if isinstance(value, bool):
        return SWITCH_ON if value else SWITCH_OFF
    text = normalize_name(value)
    if text in _PRESS_WORDS:
        return SWITCH_PRESS
    if text in _TOGGLE_WORDS:
        return SWITCH_TOGGLE
    if text in _ON_WORDS:
        return SWITCH_ON
    if text in _OFF_WORDS:
        return SWITCH_OFF
    raise DeviceError(
        f"unclear action={value!r}: expected 'on', 'off', 'press' or 'toggle'"
    )


def parse_brightness(value: Any) -> int:
    """Return an int in ``1..100`` for a ``brightness`` argument."""

    if value is None or (isinstance(value, str) and not value.strip()):
        raise DeviceError("no brightness given")
    if isinstance(value, bool):
        raise DeviceError(f"unclear brightness={value!r}: expected a number 1..100")
    if isinstance(value, str):
        text = value.strip().rstrip("%").replace(",", ".").strip()
        try:
            number = float(text)
        except ValueError as exc:
            raise DeviceError(
                f"unclear brightness={value!r}: expected a number 1..100"
            ) from exc
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise DeviceError(f"unclear brightness={value!r}: expected a number 1..100")
    if 0.0 < number <= 1.0 and isinstance(value, float):
        # tolerate a 0..1 scalar coming from the model
        number *= 100.0
    level = int(round(number))
    return max(BRIGHTNESS_MIN, min(BRIGHTNESS_MAX, level))


def parse_color(value: Any) -> tuple[int, int, int]:
    """Return ``(r, g, b)`` 0..255 for a ``color`` argument (``"#RRGGBB"``)."""

    if value is None or (isinstance(value, str) and not value.strip()):
        raise DeviceError("no color given")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if len(parts) != 3:
            raise DeviceError(f"unclear color={value!r}: expected '#RRGGBB'")
        try:
            return tuple(max(0, min(255, int(round(float(c))))) for c in parts)  # type: ignore[return-value]
        except (TypeError, ValueError) as exc:
            raise DeviceError(f"unclear color={value!r}: expected '#RRGGBB'") from exc

    text = str(value).strip()
    lowered = text.casefold()

    hex_match = _HEX_RE.match(lowered)
    if hex_match:
        digits = hex_match.group(1)
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))

    rgb_match = _RGB_CALL_RE.match(lowered)
    if rgb_match:
        return tuple(max(0, min(255, int(part))) for part in rgb_match.groups())  # type: ignore[return-value]

    named = _NAMED_COLORS.get(normalize_name(text))
    if named is not None:
        return named

    raise DeviceError(f"unclear color={value!r}: expected '#RRGGBB'")


def scale_rgb(rgb: Sequence[int], brightness: int | None) -> tuple[int, int, int]:
    """Scale an RGB triple to ``brightness`` percent, keeping the hue.

    The colour is first normalised so its strongest channel is 255 (i.e. 100 %
    brightness means "as bright as this hue can be"), then scaled down.
    """

    red, green, blue = (max(0, min(255, int(round(float(c))))) for c in tuple(rgb)[:3])
    if brightness is None:
        return (red, green, blue)
    peak = max(red, green, blue)
    if peak == 0:
        red = green = blue = 255
        peak = 255
    factor = (brightness / 100.0) * (255.0 / peak)
    out = tuple(max(0, min(255, int(round(c * factor)))) for c in (red, green, blue))
    if max(out) == 0:
        out = (1, 1, 1)
    return out  # type: ignore[return-value]


# --- device base class -------------------------------------------------------


class Device(ABC):
    """A controllable device configured in ``cfg.client.devices`` (SPEC §6).

    Type-specific settings arrive through ``params`` (everything in the YAML
    mapping besides ``name``/``type``/``area``/``description``).
    """

    #: value of the ``type`` key in config that maps to this class
    type_name: str = "device"

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        area: str | None = None,
        description: str | None = None,
    ) -> None:
        clean_name = "" if name is None else str(name).strip()
        if not clean_name:
            raise DeviceError("the config contains a device without a name")
        self.name = clean_name
        self.area = str(area).strip() if area not in (None, "") else None
        self.description = str(description).strip() if description not in (None, "") else None
        self.params: dict[str, Any] = dict(params or {})

    # -- control surface ----------------------------------------------------

    async def set_light(
        self,
        state: str,
        brightness: int | None = None,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        """Turn a light on/off, optionally with brightness (1..100) and RGB colour."""

        raise NotImplementedError(
            f"device '{self.name}' ({self.type_name}) cannot be used as a light"
        )

    async def set_switch(self, action: str) -> None:
        """Actuate a physical switch: ``on`` / ``off`` / ``press`` / ``toggle``."""

        raise NotImplementedError(
            f"device '{self.name}' ({self.type_name}) is not a switch"
        )

    # -- helpers ------------------------------------------------------------

    def param(self, *keys: str, default: Any = None) -> Any:
        """Return the first present (non-empty) value among ``keys`` in ``params``."""

        for key in keys:
            if key in self.params:
                value = self.params[key]
                if value is not None and not (isinstance(value, str) and not value.strip()):
                    return value
        return default

    def require_param(self, *keys: str) -> Any:
        value = self.param(*keys)
        if value is None:
            raise DeviceError(
                f"device '{self.name}' ({self.type_name}): the config does not set "
                f"the '{keys[0]}' parameter"
            )
        return value

    def describe(self) -> str:
        bits = [f"{self.name} ({self.type_name})"]
        if self.area:
            bits.append(f"area: {self.area}")
        if self.description:
            bits.append(self.description)
        return ", ".join(bits)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{self.__class__.__name__} name={self.name!r} area={self.area!r}>"
