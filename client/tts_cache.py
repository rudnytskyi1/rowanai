"""Кэш заранее синтезированных фраз комнаты (ТЗ 4.8).

ТЗ 4.8 просит «кэш заранее синтезированных TTS-фраз» — потому что на комнатном
ПК никакой TTS нет, а сказать «Хаб недоступен, работаю локально» нужно именно
тогда, когда хаба нет. Значит строки приносит хаб ЗАРАНЕЕ: клиент просит
синтез при подключении (``MSG_TTS_PREFETCH``), кладёт PCM на диск этого ПК и
проигрывает его, когда связи не стало.

Честность важнее удобства: файл без строки в оглавлении не играется, а
отсутствующая фраза не превращается в тишину — вызывающий код знает, что
сказать голосом нечего, и показывает строку на экране комнаты.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Оглавление кэша лежит рядом с самими файлами.
INDEX_NAME = "index.json"


def cache_key(phrase_id: str) -> str:
    """A file-safe name for one phrase id (ids come from the client itself)."""
    text = str(phrase_id or "").strip()
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in text)[:60]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{safe or 'phrase'}-{digest}"


class PhraseCache:
    """Phrases this room can say with the hub gone."""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.directory = Path(directory)
        self._index: dict[str, dict[str, Any]] = {}
        self.load_errors = 0
        if self.enabled:
            self._read_index()

    # -- index ------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.directory / INDEX_NAME

    def _read_index(self) -> None:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("Could not read the phrase cache index (%s)", exc)
            return
        if isinstance(raw, dict):
            self._index = {str(key): value for key, value in raw.items()
                           if isinstance(value, dict)}

    def _write_index(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(
                json.dumps(self._index, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not write the phrase cache index (%s)", exc)

    # -- contents ---------------------------------------------------------

    def ids(self) -> tuple[str, ...]:
        return tuple(self._index.keys())

    def has(self, phrase_id: str) -> bool:
        return str(phrase_id) in self._index

    def text_of(self, phrase_id: str) -> str:
        entry = self._index.get(str(phrase_id)) or {}
        return str(entry.get("text") or "")

    def store(self, phrase_id: str, text: str, pcm: bytes, rate: int) -> bool:
        """Keep one synthesized phrase; ``False`` when it cannot be kept."""
        if not self.enabled or not pcm:
            return False
        key = str(phrase_id or "").strip()
        if not key:
            return False
        name = f"{cache_key(key)}.pcm"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / name).write_bytes(bytes(pcm))
        except OSError as exc:
            log.warning("Could not store the phrase %s (%s)", key, exc)
            return False
        self._index[key] = {"file": name, "text": str(text or ""), "rate": int(rate or 0),
                            "bytes": len(pcm)}
        self._write_index()
        log.info("Cached the phrase %s (%d bytes at %d Hz)", key, len(pcm), int(rate or 0))
        return True

    def get(self, phrase_id: str) -> tuple[bytes, int] | None:
        """``(pcm, rate)`` of one phrase, or ``None`` when it is not here."""
        if not self.enabled:
            return None
        entry = self._index.get(str(phrase_id))
        if not entry:
            return None
        name = str(entry.get("file") or "")
        if not name:
            return None
        try:
            pcm = (self.directory / name).read_bytes()
        except OSError as exc:
            self.load_errors += 1
            log.warning("The cached phrase %s is not readable (%s)", phrase_id, exc)
            return None
        if not pcm:
            return None
        return pcm, int(entry.get("rate") or 0)


__all__ = ["INDEX_NAME", "PhraseCache", "cache_key"]
