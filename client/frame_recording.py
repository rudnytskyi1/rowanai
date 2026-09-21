"""Bounded camera writer: native-size JPEGs, no face-recognition dependency."""
import logging
from pathlib import Path
import queue
import threading

from common.recording import MediaArchive

log = logging.getLogger(__name__)


class FrameRecorder:
    def __init__(self, cfg, cv2, root=None):
        self.archive = MediaArchive.from_config(root or Path(__file__).resolve().parents[1] / 'data/camera_frames', cfg)
        self.cv2 = cv2
        self._queue = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self.saved = self.failed = 0
        self._thread = threading.Thread(target=self._run, name='rowan-camera-recording', daemon=True)
        self._thread.start()

    def submit(self, frame, captured_at, metadata, *, stop_event=None):
        # Backpressure preserves each detected-person frame if the disk is
        # slower than inference. Never silently discard frames or grow memory.
        while not self._stop.is_set() and not (stop_event and stop_event.is_set()):
            try:
                self._queue.put((frame, captured_at, metadata), timeout=.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    frame, captured_at, metadata = self._queue.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    ok, encoded = self.cv2.imencode('.jpg', frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 90])
                    if not ok:
                        raise RuntimeError('JPEG encoding failed')
                    self.archive.save(encoded.tobytes(), '.jpg', metadata, captured_at=captured_at)
                    self.saved += 1
                except Exception:
                    self.failed += 1
                    if self.failed == 1 or self.failed % 100 == 0:
                        log.exception('Camera archive failed to save a frame (%d failures)', self.failed)
                finally:
                    self._queue.task_done()
        finally:
            self.archive.close()

    def stats(self):
        return {'saved': self.saved, 'failed': self.failed, 'queued': self._queue.qsize()}

    def close(self):
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            log.warning('Camera archive is still draining %d frames', self._queue.qsize())
