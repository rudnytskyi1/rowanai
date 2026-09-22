"""Wake-word detection with Vosk in grammar mode (SPEC §7).

A small Vosk model is loaded with a restricted grammar containing only the
configured phrases plus ``"[unk]"``, which makes short-word spotting both fast
and reliable on CPU. Frames are the same 30 ms mono int16 blocks used
everywhere else in the client.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+", re.UNICODE)


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation, pad with spaces for word-boundary matching."""
    cleaned = _PUNCT_RE.sub(" ", (text or "").lower())
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return f" {cleaned} "


def _load_vosk():
    """Import Vosk lazily so the rest of the client can be imported without it."""
    try:
        import vosk  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "The vosk package is not installed. Install the client "
            "dependencies: pip install -r client/requirements.txt"
        ) from exc
    return vosk


class WakeWordDetector:
    """Feed 30 ms frames; :meth:`accept_frame` returns True when a phrase is heard."""

    def __init__(
        self,
        model_path: str | Path,
        phrases: Sequence[str],
        sample_rate: int = 16000,
        log_level: int = -1,
    ) -> None:
        path = Path(model_path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"Vosk model not found: {path}. Download it with "
                f"scripts/download-models.ps1 or fix client.wakeword.vosk_model in config.yaml"
            )
        cleaned = [p.strip().lower() for p in (phrases or []) if p and p.strip()]
        if not cleaned:
            raise ValueError("The wake-word phrase list is empty (client.wakeword.phrases/word)")
        # Keep the order, drop duplicates.
        self.phrases: list[str] = list(dict.fromkeys(cleaned))
        self._needles = [_normalize(p) for p in self.phrases]
        self.sample_rate = int(sample_rate)
        self.model_path = path

        vosk = _load_vosk()
        try:
            vosk.SetLogLevel(int(log_level))
        except Exception as exc:  # pragma: no cover - older vosk builds
            log.debug("Could not set the Vosk log level: %s", exc)
        self._vosk = vosk
        self._grammar = json.dumps([*self.phrases, "[unk]"], ensure_ascii=False)
        log.info("Loading the Vosk model: %s", path)
        self._model = vosk.Model(str(path))
        self._rec = self._new_recognizer()
        self._last_partial = ""
        log.info("Wake word active, phrases: %s", ", ".join(self.phrases))

    # -- internals -------------------------------------------------------
    def _new_recognizer(self):
        rec = self._vosk.KaldiRecognizer(self._model, float(self.sample_rate), self._grammar)
        try:
            rec.SetWords(False)
        except Exception as exc:  # pragma: no cover - older vosk builds
            log.debug("KaldiRecognizer.SetWords is unavailable: %s", exc)
        return rec

    def _match(self, text: str) -> str | None:
        if not text:
            return None
        haystack = _normalize(text)
        for phrase, needle in zip(self.phrases, self._needles):
            if needle in haystack:
                return phrase
        return None

    @staticmethod
    def _field(payload: str, key: str) -> str:
        try:
            data = json.loads(payload or "{}")
        except (ValueError, TypeError):
            return ""
        value = data.get(key, "")
        return value if isinstance(value, str) else ""

    # -- public API ------------------------------------------------------
    def accept_frame(self, frame: bytes) -> bool:
        """Push one audio frame; True means the wake word has just been heard."""
        if not frame:
            return False
        try:
            final = self._rec.AcceptWaveform(frame)
        except Exception as exc:  # pragma: no cover - native errors
            log.warning("Wake-word recogniser error: %s", exc)
            self.reset()
            return False

        if final:
            text = self._field(self._rec.Result(), "text")
            self._last_partial = ""
            hit = self._match(text)
            if hit:
                log.debug("Wake word (final): %r", text)
                return True
            return False

        partial = self._field(self._rec.PartialResult(), "partial")
        if partial and partial != self._last_partial:
            self._last_partial = partial
            hit = self._match(partial)
            if hit:
                log.debug("Wake word (partial): %r", partial)
                return True
        return False

    def accept_frames(self, frames: Iterable[bytes]) -> bool:
        """Convenience helper: True if any of the frames triggers detection."""
        for frame in frames:
            if self.accept_frame(frame):
                return True
        return False

    def reset(self) -> None:
        """Forget everything heard so far (call after each detection)."""
        self._last_partial = ""
        try:
            self._rec.Reset()
        except Exception as exc:  # pragma: no cover - older vosk builds
            log.debug("KaldiRecognizer.Reset is unavailable (%s), rebuilding the recogniser", exc)
            self._rec = self._new_recognizer()
