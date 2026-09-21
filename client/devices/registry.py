"""Device registry: builds ``{name: Device}`` from ``cfg.client.devices`` (SPEC §8).

``build_registry(cfg_client)`` is the only entry point the client core needs; the
returned :class:`DeviceRegistry` exposes ``get(name) -> Device | None`` used by the
action dispatcher.

A broken device entry (unknown ``type``, missing required param) is logged and
skipped — one bad line in ``config.yaml`` must not stop the whole assistant.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, Mapping

from .base import Device, DeviceError, normalize_name
from .magichome import MagicHomeDevice
from .switchbot import SwitchBotDevice
from .tuya import TuyaDevice

log = logging.getLogger(__name__)

#: ``type`` value in config -> driver class (SPEC §6)
DEVICE_TYPES: dict[str, type[Device]] = {
    MagicHomeDevice.type_name: MagicHomeDevice,
    TuyaDevice.type_name: TuyaDevice,
    SwitchBotDevice.type_name: SwitchBotDevice,
}

#: keys handled by ``DeviceConfig`` itself; everything else belongs to ``params``
_BASE_FIELDS = ("name", "type", "area", "description", "params")


#: A fuzzy match must be at least this long — "light" may match "kitchen light",
#: but a two-letter fragment must not match anything.
_MIN_FUZZY_LEN = 3


def _words_match(asked: str, stored: str) -> bool:
    """True if one normalised name contains the other as a whole word sequence."""

    asked_words = asked.split()
    stored_words = stored.split()
    if not asked_words or not stored_words:
        return False
    shorter, longer = sorted((asked_words, stored_words), key=len)
    if len("".join(shorter)) < _MIN_FUZZY_LEN:
        return False
    span = len(shorter)
    return any(
        longer[start : start + span] == shorter
        for start in range(len(longer) - span + 1)
    )


class DeviceRegistry:
    """Case-insensitive name -> :class:`Device` map."""

    def __init__(self, devices: Iterable[Device] = ()) -> None:
        self._devices: dict[str, Device] = {}
        for device in devices:
            self.add(device)

    # -- building -----------------------------------------------------------

    def add(self, device: Device) -> None:
        key = normalize_name(device.name)
        if not key:
            raise DeviceError("a device without a name cannot be added to the registry")
        if key in self._devices:
            log.warning(
                "device '%s' is declared several times in the config — keeping the first",
                device.name,
            )
            return
        self._devices[key] = device

    # -- lookup -------------------------------------------------------------

    def get(self, name: Any) -> Device | None:
        """Return the device by (normalised) name, or ``None`` if unknown."""

        key = normalize_name(name)
        if not key:
            return None
        device = self._devices.get(key)
        if device is not None:
            return device
        # tolerate small mismatches from speech recognition ("the led strip")
        compact = key.replace(" ", "")
        for stored_key, stored in self._devices.items():
            if compact == stored_key.replace(" ", ""):
                return stored
        # Whole-word match only ("led strip in the room" -> "led strip"). A loose substring
        # match would actuate the wrong device for a name that is not configured
        # at all, while SPEC §8 requires the dispatcher to answer "unknown device".
        matches = [
            stored
            for stored_key, stored in self._devices.items()
            if _words_match(key, stored_key)
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            log.warning(
                "the name '%s' matches several devices (%s) — refusing to guess",
                name,
                ", ".join(device.name for device in matches),
            )
        return None

    def all(self) -> list[Device]:
        return list(self._devices.values())

    @property
    def names(self) -> list[str]:
        return [device.name for device in self._devices.values()]

    def __iter__(self) -> Iterator[Device]:
        return iter(self._devices.values())

    def __len__(self) -> int:
        return len(self._devices)

    def __contains__(self, name: Any) -> bool:
        return self.get(name) is not None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<DeviceRegistry devices={self.names!r}>"


# --- config plumbing ---------------------------------------------------------


def _field(entry: Any, name: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name)
    return getattr(entry, name, None)


def _params_of(entry: Any) -> dict[str, Any]:
    """Extract the type-specific fields of a ``DeviceConfig`` (SPEC §6)."""

    params: dict[str, Any] = {}
    if isinstance(entry, Mapping):
        return {str(k): v for k, v in entry.items() if str(k) not in _BASE_FIELDS}

    extra = getattr(entry, "model_extra", None)
    if isinstance(extra, Mapping):
        params.update({str(k): v for k, v in extra.items() if str(k) not in _BASE_FIELDS})

    own = getattr(entry, "params", None)
    if isinstance(own, Mapping):
        params.update({str(k): v for k, v in own.items()})

    if not params:
        dumper = getattr(entry, "model_dump", None)
        data: Any = None
        if callable(dumper):
            try:
                data = dumper()
            except Exception as exc:  # noqa: BLE001 - fall back to __dict__
                log.debug("could not serialise the device entry: %s", exc)
        if not isinstance(data, Mapping):
            data = getattr(entry, "__dict__", None)
        if isinstance(data, Mapping):
            params.update(
                {
                    str(k): v
                    for k, v in data.items()
                    if str(k) not in _BASE_FIELDS and not str(k).startswith("_")
                }
            )
    return params


def _devices_of(cfg_client: Any) -> list[Any]:
    entries = _field(cfg_client, "devices")
    if entries is None and isinstance(cfg_client, (list, tuple)):
        entries = cfg_client
    if entries is None:
        return []
    if isinstance(entries, (list, tuple)):
        return list(entries)
    log.warning("client.devices has an unexpected type %s — ignoring it", type(entries))
    return []


def build_device(entry: Any) -> Device:
    """Build a single :class:`Device` from a ``DeviceConfig`` entry."""

    name = _field(entry, "name")
    raw_type = _field(entry, "type")
    type_key = "" if raw_type is None else str(raw_type).strip().casefold()
    if not type_key:
        raise DeviceError(f"device '{name}': no type given")
    driver = DEVICE_TYPES.get(type_key)
    if driver is None:
        known = ", ".join(sorted(DEVICE_TYPES))
        raise DeviceError(
            f"device '{name}': unknown type '{raw_type}' (known: {known})"
        )
    return driver(
        name=name,
        params=_params_of(entry),
        area=_field(entry, "area"),
        description=_field(entry, "description"),
    )


def build_registry(cfg_client: Any) -> DeviceRegistry:
    """Build the registry from ``cfg.client`` (SPEC §6/§8)."""

    registry = DeviceRegistry()
    for entry in _devices_of(cfg_client):
        try:
            device = build_device(entry)
        except DeviceError as exc:
            log.error("skipping a device from the config: %s", exc)
            continue
        except Exception as exc:  # noqa: BLE001 - a bad entry must not kill startup
            log.error(
                "skipping device '%s' from the config: %s", _field(entry, "name"), exc
            )
            continue
        try:
            registry.add(device)
        except DeviceError as exc:
            log.error("skipping a device from the config: %s", exc)
    if registry:
        log.info(
            "devices loaded (%d): %s",
            len(registry),
            "; ".join(device.describe() for device in registry),
        )
    else:
        log.info("no physical devices configured (client.devices is empty)")
    return registry
