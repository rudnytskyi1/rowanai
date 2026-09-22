"""Action dispatcher — executes the ``actions`` items sent by the server (SPEC §8).

The client main loop receives ``{"type": "actions", "items": [{"id", "tool", "args"}]}``
(SPEC §4) and feeds every item to :meth:`Dispatcher.execute`::

    dispatcher = Dispatcher(cfg.client, build_registry(cfg.client))
    ok, error, output = await dispatcher.execute(item)

The third element is the data the LLM gets back in ``action_result.output``
(SPEC §4/§8): ``run_command`` output, app-resolver hints, ``None`` otherwise.

Routing (SPEC §5): ``set_light`` / ``set_switch`` go to the device registry,
``pc_control`` (including v1.2's ``minimize_app``) and ``run_command`` to
:mod:`client.actions.pc`, and ``mouse_click`` — the action the server's
``click_screen`` tool produces — to :meth:`PCController.mouse_click`.

:meth:`Dispatcher.execute` never raises: every failure becomes
``(False, "...", output_or_None)`` so the main loop can report it back via
``action_result`` and keep listening. (``asyncio.CancelledError`` is intentionally
propagated — it means the client itself is shutting the task down, not that the
action failed.)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from ..devices.base import (
    STATE_OFF,
    STATE_ON,
    SWITCH_OFF,
    SWITCH_ON,
    Device,
    DeviceError,
    parse_brightness,
    parse_color,
    parse_state,
    parse_switch_action,
)
from .app_control import AppController
from .apps import AppError
from .browser_desktop import DesktopBrowserController
from .pc import PCActionError, PCController, PCResult

log = logging.getLogger(__name__)

# --- tool names (SPEC §5); the server sends these in the action item's "tool" field
TOOL_SET_LIGHT = "set_light"
TOOL_SET_SWITCH = "set_switch"
TOOL_PC_CONTROL = "pc_control"
TOOL_RUN_COMMAND = "run_command"
#: v1.2: produced server-side by ``click_screen``, not called by the LLM directly.
TOOL_MOUSE_CLICK = "mouse_click"
TOOL_BROWSER = "browser_control"
TOOL_APP_ACTION = 'app_action'
TOOL_SAVE_PHOTO = 'save_photo_file'
TOOL_SET_WALLPAPER = 'set_wallpaper_file'
TOOLS = frozenset(
    {
        TOOL_SET_LIGHT,
        TOOL_SET_SWITCH,
        TOOL_PC_CONTROL,
        TOOL_RUN_COMMAND,
        TOOL_MOUSE_CLICK,
        TOOL_BROWSER,
        TOOL_APP_ACTION,
        TOOL_SAVE_PHOTO,
        TOOL_SET_WALLPAPER,
    }
)

#: keys of an action item (SPEC §4)
KEY_ID = "id"
KEY_TOOL = "tool"
KEY_ARGS = "args"

#: hard limits so a dead device can never stall the voice loop
DEVICE_TIMEOUT_S = 30.0
PC_TIMEOUT_S = 20.0

#: ``run_command`` has its own 30 s budget inside pc.py; this outer guard only
#: covers the thread hand-off and stays below the server's 35 s wait (SPEC §4).
RUN_COMMAND_TIMEOUT_S = 33.0


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


class Dispatcher:
    """Routes protocol action items to devices (SPEC §8) and to the PC controller."""

    def __init__(self, cfg_client: Any, registry: Any) -> None:
        self.cfg = cfg_client
        self.registry = registry
        self.pc = PCController(self._apps_of(cfg_client))
        self.browser = DesktopBrowserController()
        self.applications = AppController(self.pc.apps)

    # -- construction helpers ----------------------------------------------

    @staticmethod
    def _apps_of(cfg_client: Any) -> dict[str, Any]:
        apps: Any = None
        if isinstance(cfg_client, Mapping):
            apps = cfg_client.get("apps")
        else:
            apps = getattr(cfg_client, "apps", None)
        if isinstance(apps, Mapping):
            return dict(apps)
        if apps:
            log.warning("client.apps has an unexpected type %s — ignoring it", type(apps))
        return {}

    # -- lifecycle ----------------------------------------------------------

    async def prepare(self) -> None:
        """Warm up the installed-app index so the first ``open_app`` is fast."""

        try:
            await self.pc.prepare()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - indexing must never break startup
            log.warning("could not build the installed-app index: %s", exc)

    # -- entry point --------------------------------------------------------

    async def execute(self, action: dict) -> tuple[bool, str | None, str | None]:
        """Execute one action item. Returns ``(ok, error, output)``; never raises."""

        action_id = "?"
        tool = "?"
        try:
            if not isinstance(action, Mapping):
                return False, f"malformed action: {action!r}", None
            action_id = str(action.get(KEY_ID) or "?")
            raw_tool = action.get(KEY_TOOL)
            tool = str(raw_tool).strip() if raw_tool is not None else ""
            raw_args = action.get(KEY_ARGS) or {}
            if not isinstance(raw_args, Mapping):
                return False, f"malformed args in action {action_id}: {raw_args!r}", None
            args: dict[str, Any] = dict(raw_args)

            log.info("action %s: %s %s", action_id, tool or "?", {k: '<image omitted>' if k in {'jpeg_base64', 'image_base64'} else v for k, v in args.items()})

            output: str | None = None
            if tool == TOOL_SET_LIGHT:
                detail = await asyncio.wait_for(
                    self._set_light(args), timeout=DEVICE_TIMEOUT_S
                )
            elif tool == TOOL_SET_SWITCH:
                detail = await asyncio.wait_for(
                    self._set_switch(args), timeout=DEVICE_TIMEOUT_S
                )
            elif tool == TOOL_PC_CONTROL:
                result = await asyncio.wait_for(
                    self._pc_control(args), timeout=PC_TIMEOUT_S
                )
                detail, output = result.detail, result.output
            elif tool == TOOL_MOUSE_CLICK:
                result = await asyncio.wait_for(
                    self._mouse_click(args), timeout=PC_TIMEOUT_S
                )
                detail, output = result.detail, result.output
            elif tool == TOOL_BROWSER:
                output = await asyncio.wait_for(self.browser.execute(args), timeout=28)
                detail = 'Browser action complete'
            elif tool == TOOL_APP_ACTION:
                import json
                result = await asyncio.wait_for(self.applications.execute(args), timeout=20)
                output, detail = json.dumps(result), 'Application inventory/action completed'
            elif tool == TOOL_SAVE_PHOTO:
                import json

                from .photos import save_photo
                result = await asyncio.wait_for(asyncio.to_thread(save_photo, args), timeout=20)
                output, detail = json.dumps(result), 'Photo saved'
            elif tool == TOOL_SET_WALLPAPER:
                import json

                from .wallpaper import set_wallpaper
                result = await asyncio.wait_for(asyncio.to_thread(set_wallpaper, args), timeout=20)
                output, detail = json.dumps(result), 'Desktop wallpaper applied and verified'
            elif tool == TOOL_RUN_COMMAND:
                ok, error, output = await asyncio.wait_for(
                    self._run_command(args), timeout=RUN_COMMAND_TIMEOUT_S
                )
                log.info(
                    "action %s finished: %s",
                    action_id,
                    "ok" if ok else f"failed ({error})",
                )
                return ok, error, output
            else:
                known = ", ".join(sorted(TOOLS))
                return False, f"unknown tool '{tool}' (known: {known})", None

            log.info("action %s finished: %s", action_id, detail or "ok")
            return True, None, output

        except asyncio.CancelledError:
            raise
        except TimeoutError:
            message = f"action {action_id} ({tool}) timed out"
            log.error(message)
            return False, message, None
        except PCActionError as exc:
            message = _error_text(exc)
            log.error("action %s (%s) failed: %s", action_id, tool, message)
            return False, message, exc.output
        except (DeviceError, AppError, NotImplementedError, ValueError) as exc:
            message = _error_text(exc)
            log.error("action %s (%s) failed: %s", action_id, tool, message)
            return False, message, None
        except Exception as exc:  # noqa: BLE001 - the dispatcher must never raise
            message = f"{exc.__class__.__name__}: {_error_text(exc)}"
            log.exception("action %s (%s) crashed: %s", action_id, tool, message)
            return False, message, None

    # -- tools --------------------------------------------------------------

    def _device(self, name: Any) -> Device:
        device = self.registry.get(name) if self.registry is not None else None
        if device is None:
            known = ""
            names = getattr(self.registry, "names", None)
            if isinstance(names, (list, tuple)) and names:
                known = f" (known: {', '.join(str(n) for n in names)})"
            raise DeviceError(f"unknown device '{name}'{known}")
        return device

    async def _set_light(self, args: dict[str, Any]) -> str:
        device = self._device(args.get("device"))

        brightness = (
            parse_brightness(args["brightness"])
            if args.get("brightness") is not None
            else None
        )
        color = parse_color(args["color"]) if args.get("color") is not None else None
        default_state = STATE_ON if (brightness is not None or color is not None) else None
        state = parse_state(args.get("state"), default=default_state)

        try:
            await device.set_light(state, brightness=brightness, color=color)
        except NotImplementedError:
            # a switch-only device (SwitchBot Bot) asked to act as a light
            if state in (STATE_ON, STATE_OFF):
                log.info(
                    "device '%s' is not a light — actuating it as a switch (%s)",
                    device.name,
                    state,
                )
                await device.set_switch(state)
            else:
                raise
        return (
            f"{device.name}: {state}"
            + (f", brightness {brightness}" if brightness is not None else "")
            + (f", color #{color[0]:02X}{color[1]:02X}{color[2]:02X}" if color else "")
        )

    async def _set_switch(self, args: dict[str, Any]) -> str:
        device = self._device(args.get("device"))
        action = parse_switch_action(args.get("action"))

        try:
            await device.set_switch(action)
        except NotImplementedError:
            # a light asked to act as a switch — on/off map cleanly
            if action in (SWITCH_ON, SWITCH_OFF):
                log.info(
                    "device '%s' is not a switch — driving it as a light (%s)",
                    device.name,
                    action,
                )
                await device.set_light(STATE_ON if action == SWITCH_ON else STATE_OFF)
            else:
                raise
        return f"{device.name}: {action}"

    async def _pc_control(self, args: dict[str, Any]) -> PCResult:
        command = args.get("command")
        value = args.get("value")
        result = await self.pc.execute(command, value)
        if not result.detail:
            return PCResult(f"pc_control: {command}", result.output)
        return result

    async def _mouse_click(self, args: dict[str, Any]) -> PCResult:
        """Click a point the server derived from the screenshot (SPEC §5 tool 6)."""

        return await self.pc.mouse_click(
            args.get("x_norm"), args.get("y_norm"), args.get("button")
        )

    async def _run_command(self, args: dict[str, Any]) -> tuple[bool, str | None, str | None]:
        return await self.pc.run_command(args.get("command"))
