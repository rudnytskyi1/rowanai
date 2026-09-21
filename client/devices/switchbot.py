"""SwitchBot Bot (button pusher) over raw BLE GATT with ``bleak`` (SPEC §8).

No cloud, no ``pySwitchbot`` — just a GATT write to the SwitchBot command
characteristic, which works on the Windows (WinRT) bleak backend::

    press  b"\\x57\\x01\\x00"
    on     b"\\x57\\x01\\x01"   (Bot must be in lever/switch mode)
    off    b"\\x57\\x01\\x02"   (Bot must be in lever/switch mode)

The Bot is connected on demand, the command is written, and the link is dropped
immediately (Bots only accept one connection at a time). One retry on failure:
the second attempt rescans for the MAC first and writes with a response, which
covers both "adapter lost the device" and "characteristic wants a response".

Config (``cfg.client.devices`` entry)::

    - name: main light
      type: switchbot_bot
      area: room
      description: main room light (button pusher on the wall switch)
      mac: "AA:BB:CC:DD:EE:FF"
      mode: press        # press | lever

Bots with a password configured are out of scope for v1.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

from .base import (
    SWITCH_OFF,
    SWITCH_ON,
    SWITCH_PRESS,
    SWITCH_TOGGLE,
    Device,
    DeviceError,
)

log = logging.getLogger(__name__)

#: SwitchBot command characteristic
CHAR_UUID = "cba20002-224d-11e6-9fb8-0002a5d5c51b"

CMD_PRESS = b"\x57\x01\x00"
CMD_ON = b"\x57\x01\x01"
CMD_OFF = b"\x57\x01\x02"

MODE_PRESS = "press"
MODE_LEVER = "lever"

#: Both attempts together must fit the dispatcher's ``DEVICE_TIMEOUT_S`` (30 s,
#: client/actions/dispatcher.py): 8 + 1 + 5 + 8 = 22 s worst case. A larger budget
#: would get the retry cancelled mid-connect and leave the BLE link half-open.
DEFAULT_CONNECT_TIMEOUT_S = 8.0
DEFAULT_SCAN_TIMEOUT_S = 5.0
RETRY_DELAY_S = 1.0


class SwitchBotDevice(Device):
    """SwitchBot Bot attached to a wall switch."""

    type_name = "switchbot_bot"

    def __init__(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        area: str | None = None,
        description: str | None = None,
    ) -> None:
        super().__init__(name, params, area, description)
        self._mac = str(self.require_param("mac", "address", "mac_address")).strip().upper()
        mode = str(self.param("mode", default=MODE_PRESS)).strip().casefold()
        if mode in ("lever", "switch"):
            self._mode = MODE_LEVER
        elif mode in ("press", "button"):
            self._mode = MODE_PRESS
        else:
            raise DeviceError(
                f"device '{self.name}': unknown mode={mode!r} "
                f"(expected 'press' or 'lever')"
            )
        try:
            self._connect_timeout = float(
                self.param("timeout", "connect_timeout", default=DEFAULT_CONNECT_TIMEOUT_S)
            )
            self._scan_timeout = float(
                self.param("scan_timeout", default=DEFAULT_SCAN_TIMEOUT_S)
            )
        except (TypeError, ValueError) as exc:
            raise DeviceError(
                f"device '{self.name}': invalid timeout in the config"
            ) from exc
        self._lock = asyncio.Lock()

    @property
    def mode(self) -> str:
        return self._mode

    # -- public API ---------------------------------------------------------

    async def set_switch(self, action: str) -> None:
        payload, human = self._payload_for(action)
        async with self._lock:
            await self._write_with_retry(payload, human)

    # -- BLE plumbing -------------------------------------------------------

    def _payload_for(self, action: str) -> tuple[bytes, str]:
        """Map a ``set_switch`` action to a SwitchBot command byte string.

        In press mode the Bot only knows one gesture, so ``on``/``off``/``toggle``
        all become a single press (SPEC §5).
        """

        if action == SWITCH_PRESS:
            return CMD_PRESS, "press"
        if action == SWITCH_TOGGLE:
            return CMD_PRESS, "toggle (single press)"
        if action == SWITCH_ON:
            if self._mode == MODE_LEVER:
                return CMD_ON, "turn on"
            return CMD_PRESS, "turn on (single press)"
        if action == SWITCH_OFF:
            if self._mode == MODE_LEVER:
                return CMD_OFF, "turn off"
            return CMD_PRESS, "turn off (single press)"
        raise DeviceError(
            f"device '{self.name}': unclear action {action!r} "
            f"(expected 'on', 'off', 'press' or 'toggle')"
        )

    async def _write_with_retry(self, payload: bytes, human: str) -> None:
        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                await self._write_once(payload, rescan=attempt > 1, response=attempt > 1)
                log.info("switchbot %s (%s): %s done", self.name, self._mac, human)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                last_error = exc
                log.warning(
                    "switchbot %s (%s): attempt %d failed: %s",
                    self.name,
                    self._mac,
                    attempt,
                    exc,
                )
                if attempt == 1:
                    await asyncio.sleep(RETRY_DELAY_S)
        raise DeviceError(
            f"device '{self.name}' ({self._mac}): could not send the command over "
            f"BLE — {last_error}"
        )

    async def _write_once(self, payload: bytes, rescan: bool, response: bool) -> None:
        try:
            from bleak import BleakClient, BleakScanner  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise DeviceError("the bleak library is missing (pip install bleak)") from exc

        target: Any = self._mac
        if rescan:
            found = await BleakScanner.find_device_by_address(
                self._mac, timeout=self._scan_timeout
            )
            if found is None:
                raise DeviceError(
                    f"SwitchBot {self._mac} was not found while scanning "
                    f"({self._scan_timeout:.0f} s)"
                )
            target = found

        client = BleakClient(target, timeout=self._connect_timeout)
        try:
            # connect() belongs inside the try: a timeout/cancellation here would
            # otherwise leave a half-established link, and a Bot accepts only one
            # connection at a time — the next voice command would fail too.
            await client.connect()
            await client.write_gatt_char(CHAR_UUID, payload, response=response)
        finally:
            await self._disconnect(client)

    async def _disconnect(self, client: Any) -> None:
        """Always drop the link, even while the action is being cancelled."""
        try:
            # shield: if we are being cancelled, the disconnect still completes.
            await asyncio.shield(client.disconnect())
        except asyncio.CancelledError:
            log.debug("switchbot %s: the disconnect continues in the background", self.name)
            raise
        except Exception as exc:  # noqa: BLE001 - disconnect must not mask errors
            log.debug("switchbot %s: error while disconnecting: %s", self.name, exc)
