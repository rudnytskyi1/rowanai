"""События присутствия, накопленные без хаба (ТЗ 4.8).

ТЗ 4.8: «после восстановления — досылка накопленных событий присутствия». Пока
хаба нет, камера комнаты продолжает работать: она видит, что человек вошёл и
ушёл, но сказать об этом некому. Буфер держит это в памяти клиента, а после
подключения отдаёт хабу по порядку, с метками времени САМОГО КЛИЕНТА — чтобы у
дома не осталось дыры в том, что происходило, пока мозг был оффлайн.

Буфер ограничен по числу записей: комната без хаба не должна съесть память
телевизора. Единственное, что копится, — состояния присутствия (счётчики и
треки). Кадры не копятся вовсе: пятиминутной давности JPEG — это не история, а
ложь о том, как комната выглядит теперь.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

log = logging.getLogger(__name__)

#: Сколько событий копим по умолчанию; 0 выключает буфер.
DEFAULT_LIMIT = 120

#: Что именно считается «событием присутствия» и досылается хабу.
REPLAYABLE = ("camera_state", "tracks")


class PresenceBuffer:
    """Bounded queue of presence frames the hub could not receive."""

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self.limit = max(0, int(limit))
        self._rows: deque[dict[str, Any]] = deque(maxlen=self.limit or None)
        self.dropped = 0
        self.stored = 0

    def add(self, payload: Any) -> bool:
        """Keep one un-sent presence frame; ``False`` when it does not belong."""
        if not self.limit or not isinstance(payload, dict):
            return False
        kind = str(payload.get("type") or "")
        if kind not in REPLAYABLE:
            return False
        row = dict(payload)
        # ТЗ 4.8: хаб должен видеть, КОГДА это было, а не когда доехало.
        row.setdefault("ts", _clock())
        row["replay"] = True
        if len(self._rows) == self._rows.maxlen:
            self.dropped += 1
        self._rows.append(row)
        self.stored += 1
        return True

    def take(self) -> list[dict[str, Any]]:
        """Everything collected, oldest first; the buffer is emptied."""
        rows = list(self._rows)
        self._rows.clear()
        return rows

    def __len__(self) -> int:
        return len(self._rows)

    def stats(self) -> dict[str, int]:
        return {"buffered": len(self._rows), "stored": self.stored, "dropped": self.dropped}


def _clock() -> float:
    import time

    return time.time()


__all__ = ["DEFAULT_LIMIT", "REPLAYABLE", "PresenceBuffer"]
