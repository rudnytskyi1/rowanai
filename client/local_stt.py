"""Локальное распознавание речи на комнатном ПК (ТЗ 4.8).

ТЗ 4.8: «Клиент держит локальный fallback: faster-whisper small (или base) на
своём GPU/CPU». Пока хаб отвечает, распознаёт он (одна большая модель на один
RTX 5090 — это и есть смысл хаба); когда хаб пропал, комната не глохнет: свои
команды («громче», «включи свет», «кино») она распознаёт локально и выполняет
тоже локально.

Модель грузится ЛЕНИВО и молча отступает: нет пакета, нет весов, нет CUDA —
значит распознавать локально нечем, и клиент продолжает работать на локальных
фразах (``client/voice_controls.py``, F-117) вместо того, чтобы делать вид, что
он что-то понял.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: Модели, которые ТЗ 4.8 называет по именам.
ALLOWED_MODELS = ("base", "small")
#: Длиннее этого локальный транскрипт не бывает: это команда, а не лекция.
MAX_TRANSCRIPT_CHARS = 400


class LocalStt:
    """faster-whisper на самом клиенте: только для локальных команд."""

    def __init__(self, cfg: Any = None, *, model_factory: Any = None) -> None:
        self.enabled = bool(getattr(cfg, "enabled", True))
        name = str(getattr(cfg, "model", "base") or "base").strip().lower()
        self.model_name = name if name in ALLOWED_MODELS else "base"
        self.device = str(getattr(cfg, "device", "auto") or "auto").strip().lower()
        self.compute_type = str(getattr(cfg, "compute_type", "int8") or "int8")
        self.language = str(getattr(cfg, "language", "") or "")
        try:
            self.timeout_s = float(getattr(cfg, "timeout_s", 20.0) or 20.0)
        except (TypeError, ValueError):
            self.timeout_s = 20.0
        #: Test seam: a callable ``(model_name, device, compute_type) -> model``.
        self._factory = model_factory
        self._model: Any = None
        self.reason = ""

    # -- loading ----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """Load the local model once; never raises, returns success."""
        if not self.enabled:
            self.reason = "local STT is switched off (client.offline.stt.enabled)"
            return False
        if self._model is not None:
            return True
        factory = self._factory or self._load_model
        for device in self._devices():
            try:
                self._model = factory(self.model_name, device, self.compute_type)
            except Exception as exc:  # noqa: BLE001 - no local STT is survivable
                self.reason = (f"{self.model_name} on {device} could not be loaded "
                               f"({type(exc).__name__}: {exc})")
                log.info("Local STT unavailable: %s", self.reason)
                continue
            log.info("Local STT ready: faster-whisper %s on %s (%s)",
                     self.model_name, device, self.compute_type)
            self.reason = ""
            return True
        return False

    def _devices(self) -> tuple[str, ...]:
        if self.device in ("cuda", "cpu"):
            return (self.device,)
        return ("cuda", "cpu")  # auto: try the card, then fall back honestly

    @staticmethod
    def _load_model(model_name: str, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel  # heavy import, lazy on purpose

        return WhisperModel(model_name, device=device, compute_type=compute_type)

    # -- transcribing -----------------------------------------------------

    def transcribe(self, pcm: bytes, sample_rate: int) -> str | None:
        """Text of one locally recorded utterance, or ``None`` if impossible."""
        if not pcm:
            return ""
        if not self.available and not self.load():
            return None
        audio = self._floats(pcm)
        if audio is None:
            return None
        try:
            segments, _info = self._model.transcribe(
                audio, language=self.language or None, beam_size=1, vad_filter=True
            )
            text = " ".join(str(getattr(segment, "text", "") or "").strip()
                            for segment in segments).strip()
        except Exception as exc:  # noqa: BLE001 - a local miss is not a crash
            self.reason = f"local transcription failed ({type(exc).__name__}: {exc})"
            log.warning("Local STT could not transcribe the utterance: %s", exc)
            return None
        return text[:MAX_TRANSCRIPT_CHARS]

    @staticmethod
    def _floats(pcm: bytes) -> Any:
        """PCM s16le bytes -> float32 samples in [-1, 1), or ``None``."""
        try:
            import numpy as np
        except Exception as exc:  # noqa: BLE001 - the camera stack ships numpy
            log.info("Local STT has no numpy (%s)", exc)
            return None
        samples = np.frombuffer(bytes(pcm), dtype="<i2").astype("float32")
        return samples / 32768.0


__all__ = ["ALLOWED_MODELS", "MAX_TRANSCRIPT_CHARS", "LocalStt"]
