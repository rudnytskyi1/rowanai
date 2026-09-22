"""Client self-update from the hub's release tag (ТЗ 4.9).

The hub says which tag a room should run. The room then does the boring, risky
part itself: ``git fetch``, ``git checkout`` the tag, migrate its config, and
restart. An update that cannot stay up is rolled back to the tag that worked —
"cannot stay up" means the client started again inside the guard window of
``healthy_after_s`` (a fresh process that survives the window calls
:meth:`mark_healthy`, and only then does the new tag count as good).

Every external effect is injected (git, restart, clock, state file), so the
decisions above are tested without a repository and without restarting anything.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def run_git(repo: Path, *args: str) -> str:
    """The real ``git``: captured, quiet, and loud when it fails."""
    process = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True,
                             timeout=120)
    if process.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {process.stderr.strip()[:200]}")
    return process.stdout.strip()


class ClientUpdater:
    """The client half of OTA: compare, fetch, check out, restart, roll back."""

    def __init__(self, *, repo: Path | str = ".", remote: str = "origin", interval_s: float = 3600,
                 healthy_after_s: float = 60.0,
                 state_path: Path | str = "data/ota_state.json",
                 git: Callable[..., str] | None = None,
                 migrate: Callable[[], Any] | None = None,
                 restart: Callable[[], Any] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.repo = Path(repo)
        self.remote, self.interval_s = remote, float(interval_s)
        self.healthy_after_s = float(healthy_after_s)
        self.state_path = Path(state_path)
        self.git = git or (lambda *args: run_git(self.repo, *args))
        self.migrate = migrate
        self.restart = restart
        self.clock = clock

    # --- the state file -----------------------------------------------------

    def state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- what the client runs now ------------------------------------------

    def current_tag(self) -> str:
        try:
            return str(self.git("describe", "--tags", "--exact-match")).strip()
        except Exception as exc:  # noqa: BLE001 - an untagged checkout is not an error
            log.debug("The client is not on a tag (%s)", exc)
            return ""

    def due(self, now: float | None = None) -> bool:
        """True when the hourly check is due again (the hub is checked at most that often)."""
        moment = self.clock() if now is None else float(now)
        state = self.state()
        if "checked_at" not in state:
            return True
        return moment - float(state.get("checked_at") or 0) >= self.interval_s

    # --- the update ---------------------------------------------------------

    async def check(self, desired: str, *, force: bool = False) -> dict[str, Any] | None:
        """Move to ``desired`` when it differs from what is checked out."""
        tag = str(desired or "").strip()
        state = self.state()
        state["checked_at"] = self.clock()
        self.save_state(state)
        if not tag:
            return None
        current = self.current_tag()
        if tag == current and not force:
            return None
        previous = current or state.get("tag") or ""
        log.info("Updating this room from %s to %s", previous or "an untagged checkout", tag)
        try:
            self.git("fetch", self.remote, "--tags", "--prune")
            self.git("checkout", tag)
        except Exception as exc:  # noqa: BLE001 - a broken fetch leaves the room as it was
            log.warning("Could not move to %s (%s); staying on %s", tag, exc, previous or "?")
            return {"action": "failed", "tag": tag, "error": str(exc)[:200]}
        if self.migrate is not None:
            try:
                self.migrate()
            except Exception as exc:  # noqa: BLE001 - migration failure is a reason to go back
                log.warning("The config migration for %s failed (%s); going back", tag, exc)
                return await self.rollback(previous, reason=f"config migration failed: {exc}")
        state = {"tag": tag, "previous": previous, "checked_at": self.clock(),
                 "pending": {"tag": tag, "previous": previous, "started_at": self.clock(),
                             "attempts": 0}}
        self.save_state(state)
        return {"action": "restart", "tag": tag, "previous": previous}

    def startup_guard(self) -> dict[str, Any] | None:
        """Run at client start: accept a healthy update or roll a broken one back."""
        state = self.state()
        pending = state.get("pending")
        if not isinstance(pending, dict):
            return None
        age = self.clock() - float(pending.get("started_at", 0))
        if age > self.healthy_after_s:
            log.info("The update to %s stayed up for %gs: keeping it", pending.get("tag"),
                     self.healthy_after_s)
            state["pending"] = None
            self.save_state(state)
            return {"action": "accepted", "tag": pending.get("tag")}
        attempts = int(pending.get("attempts", 0)) + 1
        if attempts >= 2:
            # The freshly checked-out tag did not survive the guard window.
            log.warning("Release %s did not stay up; rolling back to %s", pending.get("tag"),
                        pending.get("previous") or "the previous tag")
            previous = str(pending.get("previous") or "")
            state["pending"] = None
            self.save_state(state)
            return self._rollback_to(previous, reason="the new release did not start")
        pending["attempts"] = attempts
        state["pending"] = pending
        self.save_state(state)
        return {"action": "starting", "tag": pending.get("tag"), "attempt": attempts}

    async def rollback(self, tag: str, *, reason: str) -> dict[str, Any]:
        """Go back to ``tag`` right now (used when the new release is unusable)."""
        return self._rollback_to(tag, reason=reason)

    def _rollback_to(self, tag: str, *, reason: str) -> dict[str, Any]:
        if not tag:
            return {"action": "failed", "error": "there is no tag to go back to", "reason": reason}
        try:
            self.git("checkout", tag)
        except Exception as exc:  # noqa: BLE001 - report, never crash the client
            return {"action": "failed", "error": str(exc)[:200], "reason": reason}
        state = self.state()
        state.update({"tag": tag, "pending": None, "checked_at": self.clock()})
        self.save_state(state)
        log.info("Rolled back to %s (%s)", tag, reason)
        return {"action": "rollback", "tag": tag, "reason": reason}

    def mark_healthy(self) -> None:
        """The release has lived past the guard window; stop watching it."""
        state = self.state()
        if isinstance(state.get("pending"), dict):
            state["tag"] = state["pending"].get("tag") or state.get("tag")
            state["pending"] = None
            state["checked_at"] = self.clock()
            self.save_state(state)


__all__ = ["ClientUpdater", "run_git"]
