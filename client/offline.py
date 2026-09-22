"""Жизнь комнаты без хаба (ТЗ 4.8).

ТЗ 4.8: «Клиент держит локальный fallback: faster-whisper small (или base) на
своём GPU/CPU, локальные команды (свет, громкость, приложения, сцены), кэш
заранее синтезированных TTS-фраз. Потеря соединения дольше 3 с: HUD показывает
«мозг оффлайн», ассистент говорит «Хаб недоступен, работаю локально»; реконнект
с экспоненциальной задержкой; после восстановления — досылка накопленных
событий присутствия».

Этот модуль — только ПРАВИЛА: когда комната считается оффлайн, что именно
говорится, с какой задержкой стучится клиент и какие строки нужно иметь
заранее синтезированными. Ни звука, ни сети здесь нет, поэтому правила
проверяются часами, а не стендом.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

#: ТЗ 4.8: «потеря соединения дольше 3 с» — тогда экран говорит «мозг оффлайн».
OFFLINE_AFTER_S = 3.0
#: Экспоненциальная задержка реконнекта: первая попытка, множитель и потолок.
BACKOFF_BASE_S = 1.0
BACKOFF_FACTOR = 2.0
BACKOFF_MAX_S = 30.0

#: Строка, которую комната слышит, когда хаб ушёл. Ключи — те же id, что
#: клиент просит синтезировать заранее (``MSG_TTS_PREFETCH``, ТЗ 4.8).
OFFLINE_PHRASE_ID = "offline"
OFFLINE_PHRASES = {
    "ru": "Хаб недоступен, работаю локально.",
    "en": "The hub is unreachable. I am working locally.",
    "es": "El concentrador no responde. Trabajo en local.",
}


def language_of(value: Any) -> str:
    """Map a language name/prefix to ru/en/es (Spanish and English otherwise)."""
    text = str(value or "").strip().lower()
    if text.startswith(("ru", "рус")):
        return "ru"
    if text.startswith(("es", "spa", "esp")):
        return "es"
    return "en"


def offline_notice(language: Any = "") -> str:
    return OFFLINE_PHRASES[language_of(language)]


def prefetch_phrases() -> list[dict[str, str]]:
    """The fixed lines the room must be able to say with the hub gone."""
    return [{"id": f"{OFFLINE_PHRASE_ID}.{code}", "text": text}
            for code, text in OFFLINE_PHRASES.items()]


def backoff_delay(attempt: int, *, base_s: float = BACKOFF_BASE_S,
                  factor: float = BACKOFF_FACTOR, max_s: float = BACKOFF_MAX_S) -> float:
    """Seconds to wait before retry number ``attempt`` (1-based), capped.

    Exponential, not linear: a hub that is being restarted is back in seconds,
    while a hub that is off for the evening must not be hammered by every room
    every three seconds. ``base_s`` keeps the first retry as quick as before.
    """
    step = max(1, int(attempt))
    try:
        delay = float(base_s) * (float(factor) ** (step - 1))
    except OverflowError:  # a very long outage must not overflow the float
        delay = float(max_s)
    return max(0.1, min(float(max_s), delay))


class OfflineMode:
    """When the link is broken, what the room says and how it comes back."""

    def __init__(self, cfg: Any = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.after_s = float(getattr(cfg, "after_s", OFFLINE_AFTER_S) or OFFLINE_AFTER_S)
        self.base_s = float(getattr(cfg, "backoff_base_s", BACKOFF_BASE_S) or BACKOFF_BASE_S)
        self.factor = float(getattr(cfg, "backoff_factor", BACKOFF_FACTOR) or BACKOFF_FACTOR)
        self.max_s = float(getattr(cfg, "backoff_max_s", BACKOFF_MAX_S) or BACKOFF_MAX_S)
        self._clock = clock
        #: When the link was last known to be broken, and whether the room has
        #: already been told about THIS outage.
        self._down_since: float | None = None
        self._announced = False
        self._attempt = 0

    # -- link state -------------------------------------------------------

    def link_up(self) -> None:
        """The hub answered: the outage (if any) is over."""
        self._down_since = None
        self._announced = False
        self._attempt = 0

    def link_down(self, *, at: float | None = None) -> None:
        """Note that the link is broken; the first call starts the clock."""
        if self._down_since is None:
            self._down_since = self._now() if at is None else float(at)

    def down_s(self) -> float:
        """Seconds since the link broke (0 when it is up)."""
        if self._down_since is None:
            return 0.0
        return max(0.0, self._now() - self._down_since)

    def offline(self) -> bool:
        """True when the outage has lasted longer than ``after_s`` (ТЗ 4.8)."""
        if not self.enabled:
            return False
        return self._down_since is not None and self.down_s() >= self.after_s

    # -- what the room hears ----------------------------------------------

    def take_notice(self) -> bool:
        """``True`` once per outage, when the room must be told (ТЗ 4.8)."""
        if not self.offline() or self._announced:
            return False
        self._announced = True
        return True

    def announced(self) -> bool:
        return self._announced

    # -- coming back ------------------------------------------------------

    def next_delay(self) -> float:
        """Delay before the next reconnect attempt, and count the attempt."""
        self._attempt += 1
        return backoff_delay(self._attempt, base_s=self.base_s, factor=self.factor,
                             max_s=self.max_s)

    def delay_for(self, attempt: int) -> float:
        """The delay of attempt ``n`` (1-based) without touching the counter."""
        return backoff_delay(attempt, base_s=self.base_s, factor=self.factor,
                             max_s=self.max_s)

    def attempts(self) -> int:
        return self._attempt

    def _now(self) -> float:
        try:
            return float(self._clock())
        except Exception:  # noqa: BLE001 - a broken clock must not stop the room
            return time.monotonic()


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_FACTOR",
    "BACKOFF_MAX_S",
    "OFFLINE_AFTER_S",
    "OFFLINE_PHRASE_ID",
    "OFFLINE_PHRASES",
    "OfflineMode",
    "backoff_delay",
    "language_of",
    "offline_notice",
    "prefetch_phrases",
]
