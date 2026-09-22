"""Исполнитель computer-use на комнатном ПК (ТЗ F-512).

Хаб решает, какие шаги вообще разрешены, а этот модуль — последняя линия перед
мышью и клавиатурой: он берёт ``pyautogui`` ЛЕНИВО (нет пакета — честное
«pyautogui не установлен», а не молчаливо ничего), заново проверяет шаг по
ОБЩИМ правилам ``common.computer_use`` и только потом действует.

Две проверки, которых нет у хаба, потому что они видны только здесь:

* **активное окно.** Шаг «набрать текст» выполняет то приложение, которое
  сейчас в фокусе. Прежде чем печатать, исполнитель спрашивает у Windows имя
  активного приложения и сверяет его с allow-list; не смог узнать — отказ,
  потому что «не знаю, куда печатаю» не значит «печатай».
* **лимит шагов на самом клиенте.** Хаб мог ошибиться или устареть; счётчик
  здесь считает РЕАЛЬНО выполненные действия.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from common.computer_use import ComputerUsePolicy, ComputerUseStep, normalize_app

log = logging.getLogger(__name__)


class ComputerUseUnavailable(RuntimeError):
    """Инструмент исполнить нельзя: нет пакета или он не может работать."""


class ComputerUseRefused(RuntimeError):
    """Шаг запрещён правилами F-512 — и это не сбой, а решение."""


def foreground_app() -> str:
    """Имя приложения в активном окне Windows (``""`` — не удалось узнать).

    Активное окно даёт ``pid``; какое это приложение — знает ``tasklist`` по
    этому pid. Заголовок окна идёт в ход только как последняя подсказка,
    потому что заголовок — слова владельца, а не имя программы.
    """
    try:
        from .pc import _foreground_window

        window = _foreground_window()
    except Exception as exc:  # noqa: BLE001 - не Windows или окна нет
        log.debug("Could not read the foreground window (%s)", exc)
        return ""
    if window is None:
        return ""
    exe = _image_name_of(window.pid)
    if exe:
        return normalize_app(exe)
    title = str(getattr(window, "title", "") or "").split()
    return normalize_app(title[0]) if title else ""


def _image_name_of(pid: Any) -> str:
    """Имя exe по pid через ``tasklist``; ``""`` — не вышло."""
    try:
        from .pc import _CREATE_NO_WINDOW, TASKLIST_TIMEOUT_S, decode_console_output
    except Exception:  # noqa: BLE001 - не Windows
        return ""
    import csv
    import io
    import subprocess

    try:
        completed = subprocess.run(  # noqa: S603 - fixed command, pid is a number
            ["tasklist", "/FO", "CSV", "/NH", "/FI", f"PID eq {int(pid)}"],
            capture_output=True, creationflags=_CREATE_NO_WINDOW,
            timeout=TASKLIST_TIMEOUT_S, check=False,
        )
    except Exception as exc:  # noqa: BLE001 - tasklist мог не ответить
        log.debug("Could not ask tasklist about pid %s (%s)", pid, exc)
        return ""
    for row in csv.reader(io.StringIO(decode_console_output(completed.stdout))):
        if row and row[0].strip():
            return str(row[0])
    return ""


@dataclass
class ComputerUseExecutor:
    """Шаги агента на настоящем ПК, за ленивым импортом ``pyautogui``."""

    policy: ComputerUsePolicy
    #: Объект вроде ``pyautogui`` (в тестах — подставной: пакета в песочнице нет).
    gui: Any = None
    #: ``() -> имя активного приложения``; ``""`` — Windows не ответила.
    foreground: Callable[[], str] | None = None
    #: ``(width, height)`` экрана: координаты шага приходят ДОЛЯМИ.
    size: Callable[[], tuple[int, int]] | None = None
    #: Чем открывать приложение (по умолчанию — ``os.startfile`` на Windows).
    launcher: Callable[[str], Any] | None = None
    #: Сколько шагов РЕАЛЬНО выполнено в этом прогоне.
    performed: list[ComputerUseStep] = field(default_factory=list)

    # --- ленивый pyautogui ---------------------------------------------
    def _gui(self) -> Any:
        if self.gui is not None:
            return self.gui
        try:
            import pyautogui  # type: ignore
        except ImportError as exc:
            raise ComputerUseUnavailable(
                "pyautogui is not installed on the room PC (pip install pyautogui)"
            ) from exc
        self.gui = pyautogui
        log.info("computer use is up: pyautogui %s", getattr(pyautogui, "__version__", "?"))
        return self.gui

    def _active_app(self) -> str:
        if self.foreground is None:
            return ""
        try:
            return normalize_app(self.foreground())
        except Exception as exc:  # noqa: BLE001 - окно узнать не вышло
            log.debug("Could not read the foreground window (%s)", exc)
            return ""

    def _screen_size(self, gui: Any) -> tuple[int, int]:
        if self.size is not None:
            width, height = self.size()
            return int(width), int(height)
        size = getattr(gui, "size", None)
        if callable(size):
            size = size()
        if isinstance(size, Mapping):
            return int(size.get("width") or 0), int(size.get("height") or 0)
        try:
            width, height = size  # type: ignore[misc]
            return int(width), int(height)
        except (TypeError, ValueError):
            raise ComputerUseUnavailable("the screen size is unknown") from None

    def _open_app(self, name: str) -> None:
        if self.launcher is not None:
            self.launcher(name)
            return
        if not hasattr(os, "startfile"):  # pragma: no cover - только Windows
            raise ComputerUseUnavailable(
                f"opening {name!r} needs a launcher on this system")
        os.startfile(name)  # type: ignore[attr-defined]  # noqa: S606 - имя из allow-list

    # --- шаг ------------------------------------------------------------
    def execute(self, payload: Mapping[str, Any] | ComputerUseStep) -> dict[str, Any]:
        """Выполнить один шаг. Возвращает отчёт; исключения — только про пакет."""
        step = payload if isinstance(payload, ComputerUseStep) else None
        if step is None:
            try:
                step = ComputerUseStep.model_validate(dict(payload))
            except Exception as exc:  # noqa: BLE001 - кривой шаг это отказ
                return {"ok": False, "reason": f"the step is not understandable ({exc})",
                        "step": step, "index": len(self.performed)}
        index = len(self.performed)
        try:
            reason = self.policy.refuse(step, index=index)
            if reason:
                return {"ok": False, "reason": reason, "step": step.describe(), "index": index}
            acting_app = self._acting_app(step)
            if acting_app and not self.policy.allows_app(acting_app):
                return {"ok": False, "index": index, "step": step.describe(),
                        "reason": (f"{acting_app!r} is in front of the screen and is not "
                                   "on the list of applications computer use may touch")}
            self._perform(step, acting_app)
        except ComputerUseUnavailable:
            raise
        except ComputerUseRefused as exc:
            return {"ok": False, "reason": str(exc), "step": step.describe(), "index": index}
        except Exception as exc:  # noqa: BLE001 - сбой шага честно называется
            log.warning("Computer-use step failed (%s): %s", step.describe(), exc)
            return {"ok": False, "index": index, "step": step.describe(),
                    "reason": f"the step failed ({type(exc).__name__}: {exc})"}
        self.performed.append(step)
        return {"ok": True, "reason": "", "index": index, "step": step.describe(),
                "app": acting_app}

    def _acting_app(self, step: ComputerUseStep) -> str:
        """Приложение, которое исполнит шаг; ``""`` — шаг его не касается.

        ``wait`` и ``app`` не зависят от фокуса. Для остальных шагов фокус
        обязателен: печатать в «неизвестно какое окно» нельзя.
        """
        if step.action in ("wait", "app"):
            return normalize_app(step.app)
        active = self._active_app()
        if not active:
            raise ComputerUseRefused(
                "I cannot tell which application is in front of the screen, so I "
                "will not touch the keyboard or the mouse")
        if step.app and normalize_app(step.app) != active:
            raise ComputerUseRefused(
                f"the step asked for {step.app!r}, but {active!r} is in front of the screen")
        return active

    def _perform(self, step: ComputerUseStep, acting_app: str) -> None:
        gui = self._gui()
        if step.action == "wait":
            time_to_wait = float(step.seconds or 0.0)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            return
        if step.action == "app":
            self._open_app(step.app)
            return
        if step.action == "click":
            width, height = self._screen_size(gui)
            if not width or not height:
                raise ComputerUseUnavailable("the screen size is unknown")
            if step.x is not None and step.y is not None:
                gui.moveTo(int(round(step.x * width)), int(round(step.y * height)),
                           duration=float(step.seconds or 0.2))
            if step.button == "right":
                gui.rightClick()
            elif step.button == "double":
                gui.doubleClick()
            else:
                gui.click()
            return
        if step.action == "type":
            gui.write(step.text)
            return
        if step.action == "key":
            keys = [part.strip() for part in str(step.key or "").split("+") if part.strip()]
            if not keys:
                raise ComputerUseRefused("a key press without a key")
            if len(keys) > 1:
                gui.hotkey(*keys)
            else:
                gui.press(keys[0])
            return
        if step.action == "scroll":
            amount = int(step.amount)
            gui.scroll(amount if step.direction != "down" else -amount)
            return
        raise ComputerUseRefused(f"unknown action {step.action!r}")


@dataclass
class ComputerUseSession:
    """Один прогон агента в комнате: политика хаба, исполнитель и «стоп».

    Прогон живёт до сообщения хаба о конце задачи или до «стоп» в комнате
    (ладонь F-306, слово «стоп»): после стопа исполнитель больше не берёт
    шагов, даже если следующий дойдёт по сети — «стоп» не обсуждается.
    """

    executor: ComputerUseExecutor
    run_id: str = ""
    stopped: bool = False
    stop_reason: str = ""
    steps: int = 0

    @classmethod
    def from_hub(cls, payload: Mapping[str, Any]) -> ComputerUseSession:
        """Собрать прогон по сообщению хаба (политика приходит ОТ хаба)."""
        raw = payload.get("policy")
        policy = ComputerUsePolicy.model_validate(dict(raw)) if isinstance(raw, Mapping) \
            else ComputerUsePolicy()
        executor = ComputerUseExecutor(policy=policy, foreground=foreground_app)
        return cls(executor=executor, run_id=str(payload.get("run_id") or ""))

    def matches(self, run_id: str) -> bool:
        return bool(run_id) and run_id == self.run_id

    def stop(self, reason: str = "stopped in the room") -> None:
        self.stopped = True
        self.stop_reason = str(reason or "stopped in the room")
        log.info("Computer use %s stopped in the room: %s", self.run_id, self.stop_reason)

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Выполнить шаг или отказать, если прогон уже остановлен."""
        if self.stopped:
            return {"ok": False, "stopped": True, "index": self.steps,
                    "reason": f"the run was stopped ({self.stop_reason})"}
        report = self.executor.execute(payload)
        if report.get("ok"):
            self.steps += 1
        return report


__all__ = [
    "ComputerUseExecutor",
    "ComputerUseRefused",
    "ComputerUseSession",
    "ComputerUseUnavailable",
    "foreground_app",
]
