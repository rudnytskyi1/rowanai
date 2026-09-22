"""Magic Home / Zengge (LEDENET) Wi-Fi LED controllers via ``flux_led`` (SPEC §8).

``flux_led.WifiLedBulb`` is a synchronous, socket based client, so every call is
executed in a worker thread with :func:`asyncio.to_thread`. A fresh connection is
opened per command (the controllers drop idle sockets) and closed afterwards.

Config (``cfg.client.devices`` entry)::

    - name: led strip
      type: magichome
      area: room
      description: LED strip behind the TV
      host: 192.168.1.50
      # optional: port (5577), timeout (8), brightness (last resort default)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from .base import STATE_OFF, Device, DeviceError, scale_rgb

log = logging.getLogger(__name__)

DEFAULT_PORT = 5577
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_RGB = (255, 255, 255)


def _call_first(obj: Any, names: tuple[str, ...], *args: Any) -> Any:
    """Call the first existing method out of ``names`` (flux_led API drift)."""

    for attr in names:
        method = getattr(obj, attr, None)
        if callable(method):
            return method(*args)
    raise DeviceError(
        "flux_led: the installed version has none of the methods " + "/".join(names)
    )


class MagicHomeDevice(Device):
    """LED strip behind a Magic Home / Zengge Wi-Fi controller."""

    type_name = "magichome"

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        area: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(name, params, area, description)
        self._host = str(self.require_param("host", "ip", "address")).strip()
        try:
            self._port = int(self.param("port", default=DEFAULT_PORT))
        except (TypeError, ValueError) as exc:
            raise DeviceError(
                f"device '{self.name}': invalid port={self.param('port')!r}"
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
            from flux_led import WifiLedBulb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise DeviceError(
                "the flux_led library is missing (pip install flux_led)"
            ) from exc

        attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
            ((self._host,), {"port": self._port, "timeout": self._timeout}),
            ((self._host,), {"port": self._port}),
            ((self._host,), {}),
        )
        last_error: Exception | None = None
        for args, kwargs in attempts:
            try:
                return WifiLedBulb(*args, **kwargs)
            except TypeError as exc:
                # older/newer flux_led signature — try a simpler one
                last_error = exc
                continue
            except Exception as exc:
                raise DeviceError(
                    f"device '{self.name}': no connection to the controller {self._host}:"
                    f"{self._port} ({exc})"
                ) from exc
        raise DeviceError(
            f"device '{self.name}': flux_led rejected the connection parameters "
            f"({last_error})"
        )

    def _current_rgb(self, bulb: Any) -> tuple[int, int, int]:
        for attr in ("update_state", "refreshState", "refresh_state"):
            method = getattr(bulb, attr, None)
            if callable(method):
                try:
                    method()
                    break
                except Exception as exc:  # noqa: BLE001 - state read is best effort
                    log.debug("magichome %s: could not refresh the state: %s", self.name, exc)
                    break
        for attr in ("getRgb", "get_rgb", "rgb"):
            getter = getattr(bulb, attr, None)
            try:
                value = getter() if callable(getter) else getter
            except Exception as exc:  # noqa: BLE001 - state read is best effort
                log.debug("magichome %s: could not read the color: %s", self.name, exc)
                continue
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    rgb = tuple(int(c) for c in value[:3])
                except (TypeError, ValueError):
                    continue
                if any(rgb):
                    return rgb  # type: ignore[return-value]
        return DEFAULT_RGB

    def _sync_set_light(
        self,
        state: str,
        brightness: int | None,
        color: tuple[int, int, int] | None,
    ) -> None:
        bulb = self._connect()
        try:
            if state == STATE_OFF:
                _call_first(bulb, ("turnOff", "turn_off"))
                log.info("magichome %s: turned off", self.name)
                return

            _call_first(bulb, ("turnOn", "turn_on"))
            if color is None and brightness is None:
                log.info("magichome %s: turned on", self.name)
                return

            base_rgb = color if color is not None else self._current_rgb(bulb)
            red, green, blue = scale_rgb(base_rgb, brightness)
            _call_first(bulb, ("setRgb", "set_rgb"), red, green, blue)
            log.info(
                "magichome %s: on, color #%02X%02X%02X, brightness %s",
                self.name,
                red,
                green,
                blue,
                brightness if brightness is not None else "unchanged",
            )
        except DeviceError:
            raise
        except Exception as exc:
            raise DeviceError(
                f"device '{self.name}' ({self._host}): flux_led error — {exc}"
            ) from exc
        finally:
            closer = getattr(bulb, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001 - closing must not mask errors
                    log.debug("magichome %s: error while closing the socket: %s", self.name, exc)
