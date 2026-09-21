"""Client-side action execution (SPEC §8).

Usage from the client main loop::

    from client.actions import Dispatcher
    from client.devices import build_registry

    dispatcher = Dispatcher(cfg.client, build_registry(cfg.client))
    await dispatcher.prepare()                      # optional: warm the app index
    ok, error, output = await dispatcher.execute(item)   # item from "actions"

``output`` is the data the server hands to the LLM as the tool result
(``run_command`` output, app-resolver hints) or ``None``.
"""

from __future__ import annotations

from .apps import AppEntry, AppError, AppIndex
from .dispatcher import (
    DEVICE_TIMEOUT_S,
    PC_TIMEOUT_S,
    RUN_COMMAND_TIMEOUT_S,
    TOOL_PC_CONTROL,
    TOOL_RUN_COMMAND,
    TOOL_SET_LIGHT,
    TOOL_SET_SWITCH,
    TOOLS,
    Dispatcher,
)
from .pc import (
    PC_COMMANDS,
    PCActionError,
    PCController,
    PCResult,
    parse_hotkey,
    parse_scroll,
)

__all__ = [
    "AppEntry",
    "AppError",
    "AppIndex",
    "DEVICE_TIMEOUT_S",
    "Dispatcher",
    "PC_COMMANDS",
    "PC_TIMEOUT_S",
    "PCActionError",
    "PCController",
    "PCResult",
    "RUN_COMMAND_TIMEOUT_S",
    "TOOLS",
    "TOOL_PC_CONTROL",
    "TOOL_RUN_COMMAND",
    "TOOL_SET_LIGHT",
    "TOOL_SET_SWITCH",
    "parse_hotkey",
    "parse_scroll",
]
