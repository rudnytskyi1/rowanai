"""Privacy-режим камеры (ТЗ F-303).

«Rowan, перестань смотреть» — это не выключение микрофона и не пауза в
разговоре: камера перестаёт отдавать кадры, в HUD загорается значок, а голос
продолжает работать, потому что иначе камеру нельзя было бы вернуть тем же
голосом («смотри снова»).

Флаг живёт на КЛИЕНТЕ: кадры уходят с этого компьютера, и обещать «хаб больше
не смотрит» может только тот, кто их не отправляет. Хаб держит свою копию
состояния, чтобы не ждать кадров и правильно отвечать на вопросы F-301, а при
подключении клиент говорит, что у него на самом деле (`hello.privacy`).

Аппаратный индикатор (USB-реле или светодиод) — P2 в ТЗ; программный (значок в
HUD) обязателен и живёт здесь.
"""
from __future__ import annotations

from typing import Any

INDICATOR: dict[str, str] = {
    "ru": "Камера выключена",
    "en": "Camera off",
    "es": "Cámara apagada",
}

ON: dict[str, str] = {
    "ru": "Хорошо, камера выключена. Кадры больше не уходят, значок в углу экрана.",
    "en": "Alright, the camera is off. No frames are sent, the icon is on screen.",
    "es": "De acuerdo, la cámara está apagada. No se envían imágenes y el icono está en pantalla.",
}

OFF: dict[str, str] = {
    "ru": "Камера снова смотрит.",
    "en": "The camera is watching again.",
    "es": "La cámara vuelve a mirar.",
}

ALREADY: dict[str, str] = {
    "ru": "Камера уже выключена — кадры и так не уходят.",
    "en": "The camera is already off - no frames are being sent.",
    "es": "La cámara ya está apagada: no se envían imágenes.",
}


def language_of(value: Any, *, default: str = "en") -> str:
    code = str(value or "").strip().casefold()[:2]
    return code if code in INDICATOR else default


class PrivacyMode:
    """The flag, the HUD caption and the one rule the camera obeys."""

    def __init__(self, language: Any = "en") -> None:
        self.language = language_of(language)
        self.on = False
        self.reason = ""

    @property
    def allows_frames(self) -> bool:
        return not self.on

    def set(self, on: bool, *, reason: str = "", language: Any = None) -> bool:
        """Apply the state; ``True`` when it really changed."""
        if language is not None:
            self.language = language_of(language, default=self.language)
        wanted = bool(on)
        changed = wanted != self.on
        self.on, self.reason = wanted, str(reason or "")[:100]
        return changed

    def auto(self) -> bool:
        """Whether the camera came back because somebody spoke (F-303 choice)."""
        return self.reason == "voice"

    def indicator(self) -> str:
        """The HUD caption, or an empty string when the camera is on."""
        return INDICATOR[self.language] if self.on else ""

    def confirmation(self, *, changed: bool) -> str:
        """What the room hears after the request (never a silent switch)."""
        if not changed and self.on:
            return ALREADY[self.language]
        return ON[self.language] if self.on else OFF[self.language]

    def summary(self) -> dict[str, Any]:
        return {"on": self.on, "reason": self.reason, "indicator": self.indicator()}


__all__ = ["ALREADY", "INDICATOR", "OFF", "ON", "PrivacyMode", "language_of"]
