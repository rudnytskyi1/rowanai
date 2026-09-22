"""Installed-application index and resolver (SPEC §8, v1.1).

``pc_control`` with ``open_app`` / ``close_app`` receives the app name exactly as
the user said it ("chrome", "calculator", "vs code"). This module turns that name
into something launchable:

1. the index is built once, off the event loop, from
   ``powershell -NoProfile -Command "Get-StartApps | ConvertTo-Json"`` — the Start
   Menu listing covers BOTH classic desktop programs and Store/UWP apps as
   ``{Name, AppID}`` pairs;
2. ``cfg.client.apps`` entries (friendly name -> exe path / command line) are
   merged on top with the highest priority;
3. :meth:`AppIndex.resolve` matches a spoken name against the index: exact
   (case-insensitive) -> prefix -> fuzzy (``difflib``, cutoff 0.6). A miss
   triggers one lazy refresh (the app may have been installed after startup) and,
   if it still misses, returns ``None`` — the caller reports the closest names
   from :meth:`AppIndex.suggestions` so the LLM can retry with a better name.

Launching: config entries go through ``os.startfile(path)`` (or ``Popen`` when the
entry carries arguments), Start Menu entries through
``os.startfile("shell:AppsFolder\\<AppID>")``, which works for desktop and UWP
apps alike.

Closing: :meth:`AppEntry.process_name` returns the image name for
``taskkill /IM`` when the entry maps to an executable; it returns ``None`` for
Store/UWP AppIDs (``PackageFamilyName!AppId``), which cannot be mapped to a
process image — the caller turns that into a clear error.
"""

from __future__ import annotations

import asyncio
import ctypes
import difflib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# --- where an index entry came from ------------------------------------------
SOURCE_CONFIG = "config"
SOURCE_START_MENU = "start_menu"

#: PowerShell snippet that dumps the Start Menu app list as JSON. The encoding
#: line makes PowerShell write UTF-8 to the pipe instead of the OEM code page.
LIST_APPS_COMMAND = (
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
)

#: shell namespace prefix used to launch a Start Menu / Store app by AppID
SHELL_APPS_FOLDER = "shell:AppsFolder\\"

#: how long the indexing PowerShell call may take
INDEX_TIMEOUT_S = 25.0

#: a miss refreshes the index at most once per this many seconds
REFRESH_COOLDOWN_S = 60.0

#: difflib cutoff for accepting a fuzzy match (SPEC §8)
FUZZY_CUTOFF = 0.6

#: looser cutoff used only to suggest alternatives in an error message
SUGGEST_CUTOFF = 0.3

#: how many close names a miss reports back to the LLM (SPEC §8)
MAX_SUGGESTIONS = 3

_IS_WINDOWS = sys.platform == "win32"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


class AppError(RuntimeError):
    """Recoverable failure while resolving or launching an application."""


# --- small shared helpers (also used by client/actions/pc.py) ----------------


def powershell_executable() -> str:
    """Return the path of ``powershell.exe`` (falls back to the system copy)."""

    found = shutil.which("powershell")
    if found:
        return found
    system_root = os.environ.get("SystemRoot") or "C:\\Windows"
    fallback = os.path.join(
        system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )
    return fallback if os.path.isfile(fallback) else "powershell"


def console_encoding() -> str:
    """Best guess for the encoding a console child process writes in."""

    if _IS_WINDOWS:
        try:
            return f"cp{ctypes.WinDLL('kernel32').GetOEMCP()}"
        except Exception:  # noqa: BLE001 - fall back to UTF-8
            pass
    return "utf-8"


def decode_console_output(data: bytes | None) -> str:
    """Decode child-process output: UTF-8 first, OEM code page as a fallback."""

    if not data:
        return ""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode(console_encoding(), errors="replace")


def expand_path(value: Any) -> str:
    """Expand ``%VARS%`` and ``~`` in a config path / command line."""

    return os.path.expandvars(os.path.expanduser(str(value).strip()))


def split_command(target: str) -> tuple[str, list[str]]:
    """Split a config entry into ``(executable, arguments)``.

    A bare path is kept intact even when it contains spaces
    (``C:\\Program Files\\...\\chrome.exe``); a real command line
    (``notepad notes.txt``) is split into program plus arguments.
    """

    text = str(target).strip()
    for candidate in (text, text.strip('"')):
        if candidate and (os.path.isfile(candidate) or os.path.isdir(candidate)):
            return candidate, []

    bare = text.strip('"')
    if bare.lower().endswith(".exe") and ("\\" in bare or "/" in bare):
        # full path to an executable that is not on disk (yet) — do not split it
        return bare, []

    try:
        parts = shlex.split(text, posix=False)
    except ValueError:
        parts = [text]
    parts = [part.strip('"') for part in parts if part.strip()]
    if not parts:
        return bare, []
    return parts[0], parts[1:]


def normalize_app_name(value: Any) -> str:
    """Normalise an app name for lookups (case, spacing, separators)."""

    text = "" if value is None else str(value)
    for separator in ("_", "-", "\t", "\n", "\r"):
        text = text.replace(separator, " ")
    return " ".join(text.split()).strip().casefold()


# --- index entries ------------------------------------------------------------


@dataclass(frozen=True)
class AppEntry:
    """One resolvable application."""

    #: display name (config key or Start Menu name)
    name: str
    #: exe path / command line (config) or Start Menu AppID
    target: str
    #: :data:`SOURCE_CONFIG` or :data:`SOURCE_START_MENU`
    source: str = SOURCE_START_MENU

    @property
    def is_config(self) -> bool:
        return self.source == SOURCE_CONFIG

    @property
    def is_store_app(self) -> bool:
        """True for a Store/UWP AppID (``PackageFamilyName!AppId``)."""

        return not self.is_config and "!" in self.target

    def process_name(self) -> str | None:
        """Image name for ``taskkill /IM``, or ``None`` if there is no process.

        Store/UWP apps run inside a shared host process that the AppID does not
        name, so they cannot be closed this way (SPEC §8).
        """

        target = str(self.target).strip().strip('"')
        if not target:
            return None
        if self.is_config:
            executable, _ = split_command(expand_path(target))
            base = os.path.basename(executable.replace("/", "\\").strip().strip('"'))
            stem, ext = os.path.splitext(base)
            if not stem:
                return None
            return base if ext.lower() == ".exe" else f"{stem}.exe"
        if "!" in target:
            return None
        base = os.path.basename(target.replace("/", "\\"))
        return base if base.lower().endswith(".exe") else None

    def launch(self) -> str:
        """Start the application (blocking). Returns what was launched."""

        if not _IS_WINDOWS:
            raise AppError("launching applications is only supported on Windows")
        if self.is_config:
            return _launch_target(self.target)
        app_id = str(self.target).strip()
        if not app_id:
            raise AppError(f"application '{self.name}' has an empty AppID")
        link = SHELL_APPS_FOLDER + app_id
        try:
            os.startfile(link)  # type: ignore[attr-defined]  # noqa: S606 - Windows launcher
        except OSError as exc:
            raise AppError(f"could not launch '{self.name}' ({link}): {exc}") from exc
        return link

    def describe(self) -> str:
        kind = "config" if self.is_config else ("store app" if self.is_store_app else "app")
        return f"{self.name} [{kind}] -> {self.target}"


def _launch_target(target: str) -> str:
    """Launch a ``cfg.client.apps`` entry (path, or command line with arguments)."""

    expanded = expand_path(target)
    if not expanded:
        raise AppError("empty launch path in client.apps")
    executable, arguments = split_command(expanded)

    if not arguments and (os.path.isfile(expanded) or os.path.isdir(expanded)):
        try:
            os.startfile(expanded)  # type: ignore[attr-defined]  # noqa: S606 - Windows launcher
        except OSError as exc:
            raise AppError(f"could not launch '{expanded}': {exc}") from exc
        return expanded

    if not executable:
        raise AppError(f"empty launch path: '{target}'")
    resolved = executable if os.path.isfile(executable) else shutil.which(executable)
    if not resolved:
        raise AppError(f"executable not found: '{executable}'")

    if not arguments:
        try:
            os.startfile(resolved)  # type: ignore[attr-defined]  # noqa: S606 - Windows launcher
        except OSError as exc:
            raise AppError(f"could not launch '{resolved}': {exc}") from exc
        return resolved

    try:
        subprocess.Popen(  # noqa: S603 - command comes from the local config file
            [resolved, *arguments],
            cwd=os.path.dirname(resolved) or None,
            close_fds=True,
            creationflags=_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP,
        )
    except OSError as exc:
        raise AppError(f"could not launch '{resolved}': {exc}") from exc
    return " ".join([resolved, *arguments])


# --- Start Menu listing (blocking, runs in a worker thread) -------------------


def _parse_start_apps(payload: str) -> list[AppEntry]:
    """Parse the JSON produced by ``Get-StartApps | ConvertTo-Json``."""

    text = payload.strip()
    if not text:
        return []
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Get-StartApps returned unparsable JSON: %s", exc)
        return []

    # ConvertTo-Json emits a bare object when the pipeline yields a single item.
    if isinstance(data, Mapping):
        data = [data]
    if not isinstance(data, list):
        log.warning("Get-StartApps returned unexpected JSON of type %s", type(data).__name__)
        return []

    entries: list[AppEntry] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("Name") or "").strip()
        app_id = str(item.get("AppID") or item.get("AppId") or "").strip()
        if not name or not app_id:
            continue
        entries.append(AppEntry(name=name, target=app_id, source=SOURCE_START_MENU))
    return entries


def list_start_apps() -> list[AppEntry]:
    """Return the Start Menu app list. Never raises — a failure yields ``[]``."""

    if not _IS_WINDOWS:
        log.warning("the installed-app index needs Windows — starting empty")
        return []
    command = [
        powershell_executable(),
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        LIST_APPS_COMMAND,
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed command, no user input
            command,
            capture_output=True,
            timeout=INDEX_TIMEOUT_S,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.error("Get-StartApps timed out after %.0f s", INDEX_TIMEOUT_S)
        return []
    except OSError as exc:
        log.error("could not run PowerShell for Get-StartApps: %s", exc)
        return []

    if completed.returncode != 0:
        detail = decode_console_output(completed.stderr).strip() or decode_console_output(
            completed.stdout
        ).strip()
        log.error(
            "Get-StartApps failed with exit code %s: %s",
            completed.returncode,
            detail or "(no output)",
        )
        return []
    return _parse_start_apps(decode_console_output(completed.stdout))


# --- the index ----------------------------------------------------------------


class AppIndex:
    """Cached index of installed applications with a fuzzy resolver (SPEC §8)."""

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        self._overrides: list[AppEntry] = []
        for key, value in dict(overrides or {}).items():
            name = str(key).strip()
            target = "" if value is None else str(value).strip()
            if not name or not target:
                log.warning("ignoring client.apps entry %r: empty name or path", key)
                continue
            self._overrides.append(AppEntry(name=name, target=target, source=SOURCE_CONFIG))

        self._entries: dict[str, AppEntry] = {}
        self._loaded = False
        #: monotonic time of the last refresh a lookup miss triggered (None = never)
        self._last_miss_refresh: float | None = None
        self._lock = asyncio.Lock()
        self._rebuild_from(())

    # -- state --------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Display names of everything in the index."""

        return [entry.name for entry in self._entries.values()]

    @property
    def entries(self):
        return list(self._entries.values())

    @property
    def loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        return len(self._entries)

    # -- building -----------------------------------------------------------

    def _rebuild_from(self, start_apps: Iterable[AppEntry]) -> None:
        """Rebuild the name map: config overrides first, Start Menu behind them."""

        entries: dict[str, AppEntry] = {}
        for entry in self._overrides:
            entries.setdefault(normalize_app_name(entry.name), entry)
        for entry in start_apps:
            entries.setdefault(normalize_app_name(entry.name), entry)
        self._entries = entries

    async def _build(self) -> int:
        """Run the indexing off the event loop and cache the result."""

        start_apps = await asyncio.to_thread(list_start_apps)
        self._rebuild_from(start_apps)
        self._loaded = True
        log.info(
            "installed-app index ready: %d entries (%d from the Start Menu, "
            "%d from client.apps)",
            len(self._entries),
            len(start_apps),
            len(self._overrides),
        )
        return len(self._entries)

    async def ensure_ready(self) -> None:
        """Build the index once; concurrent callers wait for the same build."""

        if self._loaded:
            return
        async with self._lock:
            if not self._loaded:
                await self._build()

    async def refresh(self) -> int:
        """Rebuild the index now (an app may have been installed since startup)."""

        async with self._lock:
            return await self._build()

    # -- lookup -------------------------------------------------------------

    def _match(self, query: str) -> AppEntry | None:
        """Exact -> prefix -> fuzzy match over the cached names (SPEC §8)."""

        if not query:
            return None
        exact = self._entries.get(query)
        if exact is not None:
            return exact

        prefixed = [
            (key, entry)
            for key, entry in self._entries.items()
            if key.startswith(query) or query.startswith(key)
        ]
        if prefixed:
            # the shortest name is the least surprising interpretation
            prefixed.sort(key=lambda item: (len(item[0]), item[0]))
            return prefixed[0][1]

        close = difflib.get_close_matches(query, list(self._entries), n=1, cutoff=FUZZY_CUTOFF)
        if close:
            return self._entries[close[0]]
        return None

    async def resolve(self, name: Any) -> AppEntry | None:
        """Resolve a spoken app name, refreshing the index once on a miss."""

        query = normalize_app_name(name)
        if not query:
            return None
        await self.ensure_ready()

        entry = self._match(query)
        if entry is not None:
            log.debug("app '%s' resolved to %s", name, entry.describe())
            return entry

        # The app may have been installed after the index was built: rebuild once,
        # then trust the cache again for a while so a retrying model cannot spawn
        # a PowerShell process per guess.
        now = time.monotonic()
        if self._last_miss_refresh is None or now - self._last_miss_refresh >= REFRESH_COOLDOWN_S:
            self._last_miss_refresh = now
            log.info("app '%s' is not in the index — refreshing it once", name)
            await self.refresh()
            entry = self._match(query)
            if entry is not None:
                return entry
        return None

    def suggestions(self, name: Any, limit: int = MAX_SUGGESTIONS) -> list[str]:
        """Up to ``limit`` installed names closest to ``name`` (for error text)."""

        query = normalize_app_name(name)
        if not query or not self._entries:
            return []
        keys = list(self._entries)
        close = difflib.get_close_matches(query, keys, n=limit, cutoff=SUGGEST_CUTOFF)
        if not close:
            close = [key for key in keys if query in key or key in query][:limit]
        return [self._entries[key].name for key in close]

    def miss_hint(self, name: Any, limit: int = MAX_SUGGESTIONS) -> str:
        """Human-readable hint appended to an ``unknown app`` error."""

        close = self.suggestions(name, limit)
        if close:
            return "closest installed apps: " + ", ".join(close)
        if not self._entries:
            return "the installed-app index is empty"
        return f"{len(self._entries)} apps are indexed, none of them looks similar"


__all__ = [
    "AppEntry",
    "AppError",
    "AppIndex",
    "FUZZY_CUTOFF",
    "LIST_APPS_COMMAND",
    "MAX_SUGGESTIONS",
    "REFRESH_COOLDOWN_S",
    "SHELL_APPS_FOLDER",
    "SOURCE_CONFIG",
    "SOURCE_START_MENU",
    "console_encoding",
    "decode_console_output",
    "expand_path",
    "list_start_apps",
    "normalize_app_name",
    "powershell_executable",
    "split_command",
]
