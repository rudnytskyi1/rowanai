"""Tuya / Smart Life Wi-Fi LED devices via ``tinytuya`` (SPEC §8).

``tinytuya.BulbDevice`` is synchronous (local TCP protocol), so calls run in a
worker thread through :func:`asyncio.to_thread`.

Config (``cfg.client.devices`` entry)::

    - name: garland
      type: tuya
      area: room
      description: window garland
      dev_id: "xxxxxxxxxxxx"
      host: 192.168.1.51
      local_key: "yyyyyyyyyyyy"
      version: 3.3
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from .base import STATE_OFF, Device, DeviceError

log = logging.getLogger(__name__)

DEFAULT_VERSION = 3.3
DEFAULT_TIMEOUT_S = 6.0


def _check(result: Any, what: str, device: str) -> None:
    """Raise :class:`DeviceError` if tinytuya returned an error payload."""

    if isinstance(result, Mapping):
        error = result.get("Error") or result.get("error")
        if error:
            code = result.get("Err") or result.get("err")
            suffix = f" (code {code})" if code else ""
            raise DeviceError(f"device '{device}': {what} — {error}{suffix}")


class TuyaDevice(Device):
    """Tuya Wi-Fi LED strip / bulb controlled over the local network."""

    type_name = "tuya"

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        area: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(name, params, area, description)
        self._dev_id = str(self.require_param("dev_id", "device_id", "id")).strip()
        self._host = str(self.require_param("host", "ip", "address")).strip()
        self._local_key = str(self.require_param("local_key", "key", "localkey")).strip()
        raw_version = self.param("version", "protocol_version", default=DEFAULT_VERSION)
        try:
            self._version = float(raw_version)
        except (TypeError, ValueError) as exc:
            raise DeviceError(
                f"device '{self.name}': invalid version={raw_version!r}"
            ) from exc
        try:
            self._timeout = float(self.param("timeout", default=DEFAULT_TIMEOUT_S))
        except (TypeError, ValueError) as exc:
            raise DeviceError(
                f"device '{self.name}': invalid timeout={self.param('timeout')!r}"
            ) from exc
        self._lock = asyncio.Lock()

    # -- public API ---------------------------------------------------------

    async def set_light(
        self,
        state: str,
        brightness: int | None = None,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(self._sync_set_light, state, brightness, color)

    # -- worker-thread implementation ---------------------------------------

    def _connect(self) -> Any:
        try:
            import tinytuya  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise DeviceError(
                "the tinytuya library is missing (pip install tinytuya)"
            ) from exc

        try:
            bulb = tinytuya.BulbDevice(self._dev_id, self._host, self._local_key)
            bulb.set_version(self._version)
            setter = getattr(bulb, "set_socketPersistent", None)
            if callable(setter):
                setter(False)
            timeout_setter = getattr(bulb, "set_socketTimeout", None)
            if callable(timeout_setter):
                timeout_setter(self._timeout)
        except Exception as exc:
            raise DeviceError(
                f"device '{self.name}': could not prepare the connection to "
                f"{self._host} ({exc})"
            ) from exc
        return bulb

    def _sync_set_light(
        self,
        state: str,
        brightness: int | None,
        color: tuple[int, int, int] | None,
    ) -> None:
        bulb = self._connect()
        try:
            if state == STATE_OFF:
                _check(bulb.turn_off(), "turning off", self.name)
                log.info("tuya %s: turned off", self.name)
                return

            _check(bulb.turn_on(), "turning on", self.name)
            if color is not None:
                red, green, blue = color
                _check(bulb.set_colour(red, green, blue), "setting the color", self.name)
            if brightness is not None:
                _check(
                    bulb.set_brightness_percentage(int(brightness)),
                    "setting the brightness",
                    self.name,
                )
            log.info(
                "tuya %s: on, color %s, brightness %s",
                self.name,
                "#%02X%02X%02X" % color if color is not None else "unchanged",
                brightness if brightness is not None else "unchanged",
            )
        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(
                f"device '{self.name}' ({self._host}): tinytuya error — {exc}"
            ) from exc
        finally:
            closer = getattr(bulb, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001 - closing must not mask errors
                    log.debug("tuya %s: error while closing the socket: %s", self.name, exc)
