"""Smart-home device drivers executed on the room client (SPEC §8).

Public surface::

    from client.devices import build_registry
    registry = build_registry(cfg.client)
    device = registry.get("led strip")      # Device | None
"""

from __future__ import annotations

from .base import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    LIGHT_STATES,
    STATE_OFF,
    STATE_ON,
    SWITCH_ACTIONS,
    SWITCH_OFF,
    SWITCH_ON,
    SWITCH_PRESS,
    SWITCH_TOGGLE,
    Device,
    DeviceError,
    normalize_name,
    parse_brightness,
    parse_color,
    parse_state,
    parse_switch_action,
    scale_rgb,
)
from .magichome import MagicHomeDevice
from .registry import DEVICE_TYPES, DeviceRegistry, build_device, build_registry
from .switchbot import SwitchBotDevice
from .tuya import TuyaDevice

__all__ = [
    "BRIGHTNESS_MAX",
    "BRIGHTNESS_MIN",
    "DEVICE_TYPES",
    "Device",
    "DeviceError",
    "DeviceRegistry",
    "LIGHT_STATES",
    "MagicHomeDevice",
    "STATE_OFF",
    "STATE_ON",
    "SWITCH_ACTIONS",
    "SWITCH_OFF",
    "SWITCH_ON",
    "SWITCH_PRESS",
    "SWITCH_TOGGLE",
    "SwitchBotDevice",
    "TuyaDevice",
    "build_device",
    "build_registry",
    "normalize_name",
    "parse_brightness",
    "parse_color",
    "parse_state",
    "parse_switch_action",
    "scale_rgb",
]
