"""Local, indexed recordings. Retention only touches files owned by this index."""
import io
import json
import shutil
import sqlite3
import threading
import time
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MediaArchive:
    def __init__(self, root: Path, *, retention_days: int = 0, max_gb: float = 0,
                 min_free_gb: float = 5) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_s = retention_days * 86400
        self.max_bytes = int(max_gb * 1024**3)
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.root / 'index.sqlite3', check_same_thread=False)
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.execute('''CREATE TABLE IF NOT EXISTS recordings
            (id TEXT PRIMARY KEY, captured_at REAL NOT NULL, path TEXT NOT NULL UNIQUE,
             bytes INTEGER NOT NULL, metadata TEXT NOT NULL)''')
        self._db.execute('CREATE INDEX IF NOT EXISTS recordings_time ON recordings(captured_at)')
        self._db.commit()
        row = self._db.execute('SELECT COALESCE(SUM(bytes), 0) FROM recordings').fetchone()
        self._bytes = int(row[0]) if row else 0
        self._last_prune = 0.
        self.saved = self.failures = 0

    @classmethod
    def from_config(cls, root: Path, cfg: Any) -> "MediaArchive":
        return cls(root, retention_days=cfg.retention_days, max_gb=cfg.max_gb, min_free_gb=cfg.min_free_gb)

    def _owned_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path.suffix not in ('.wav', '.jpg'):
            raise ValueError('Invalid recording path in archive index')
        return path

    def _prune(self, now: float, incoming: int) -> None:
        if not self.retention_s and not self.max_bytes:
            return
        over_budget = self.max_bytes and self._bytes + incoming > self.max_bytes
        if not over_budget and now - self._last_prune < 60:
            return
        cutoff = now - self.retention_s if self.retention_s else float('-inf')
        while True:
            rows = self._db.execute(
                'SELECT id, captured_at, path, bytes FROM recordings ORDER BY captured_at LIMIT 128'
            ).fetchall()
            if not rows:
                break
            removed = 0
            for record_id, captured, relative, size in rows:
                over_budget = self.max_bytes and self._bytes + incoming > self.max_bytes
                if captured >= cutoff and not over_budget:
                    break
                path = self._owned_path(relative)
                path.unlink(missing_ok=True)
                self._db.execute('DELETE FROM recordings WHERE id=?', (record_id,))
                self._bytes -= size
                removed += 1
                # Minute directories keep large archives browsable. Remove only
                # now-empty directories under this archive, never recursively.
                parent = path.parent
                while parent != self.root:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            self._db.commit()
            if removed < len(rows):
                break
        self._last_prune = now

    def save(self, data: bytes, extension: str, metadata: dict[str, Any], *,
             captured_at: float | None = None) -> dict[str, Any]:
        if extension not in ('.wav', '.jpg'):
            raise ValueError('Unsupported recording format')
        if self.max_bytes and len(data) > self.max_bytes:
            raise OSError('Recording exceeds the configured archive size')
        captured = time.time() if captured_at is None else captured_at
        moment = datetime.fromtimestamp(captured, UTC)
        record_id = uuid.uuid4().hex
        relative = f'{moment:%Y-%m-%d/%H/%M}/{moment:%H%M%S-%f}-{record_id[:12]}{extension}'
        with self._lock:
            self._prune(time.time(), len(data))
            if shutil.disk_usage(self.root).free < self.min_free_bytes + len(data):
                raise OSError('Recording stopped: configured free disk reserve reached')
            target = self._owned_path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_suffix(extension + '.pending')
            try:
                with pending.open('xb') as handle:
                    handle.write(data)
                pending.replace(target)
                self._db.execute('INSERT INTO recordings VALUES(?,?,?,?,?)',
                                 (record_id, captured, relative, len(data), json.dumps(metadata, ensure_ascii=False)))
                self._db.commit()
            finally:
                pending.unlink(missing_ok=True)
            self._bytes += len(data)
            self.saved += 1
        return {'id': record_id, 'path': relative, 'captured_at': moment.isoformat()}

    def save_audio(self, pcm: bytes, sample_rate: int, metadata: dict[str, Any], *,
                   captured_at: float | None = None) -> dict[str, Any]:
        output = io.BytesIO()
        with wave.open(output, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(pcm[:len(pcm) - len(pcm) % 2])
        return self.save(output.getvalue(), '.wav', metadata, captured_at=captured_at)

    def annotate(self, record_id: str, fields: dict[str, Any]) -> None:
        with self._lock:
            row = self._db.execute('SELECT metadata FROM recordings WHERE id=?', (record_id,)).fetchone()
            if row:
                metadata = json.loads(row[0])
                metadata.update(fields)
                self._db.execute('UPDATE recordings SET metadata=? WHERE id=?',
                                 (json.dumps(metadata, ensure_ascii=False), record_id))
                self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
