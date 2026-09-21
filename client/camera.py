"""Room camera service: presence, object counts and face frames (SPEC v1.4).

The C920 on the room PC is the assistant's eyes. This module reports STATE, not
video:

* an OpenCV capture thread keeps the newest frame of ``cfg.client.camera.index``
  in memory (draining the driver buffer so the frame is always fresh),
* a second thread runs Ultralytics YOLO (``cfg.client.camera.model``) over fresh
  frames, capped by ``cfg.client.camera.fps`` (0 means no software limit), and counts the labels it sees
  with a confidence of at least :data:`CONF_THRESHOLD`,
* whenever the picture changes it sends ``camera_state`` (debounced, at least
  :data:`STATE_DEBOUNCE_S` apart),
* while at least one person is visible it pushes a burst of :data:`FACE_BURST`
  JPEGs every ``cfg.client.camera.face_check_interval_s`` as ``camera_frame``
  (``reason: "presence"``, id ``p<N>``, headers carrying ``seq``/``of``) so the
  server can match faces over several frames instead of just one,
* and it answers the server's ``camera_request`` -- one frame, or a burst of
  up to :data:`common.protocol.CAMERA_BURST_MAX` when the request carries a
  ``burst`` count (``reason: "request"``) -- the way :mod:`client.screen`
  answers a ``screenshot_request``.

Every burst (pulled or pushed) is a sequence of header+binary pairs sharing one
``id``, each header numbering itself ``seq`` of ``of`` total, captured roughly
:data:`BURST_FRAME_INTERVAL_S` apart from FRESH frames of the capture thread
(never the same JPEG resent twice) and sent back to back under one hold of the
wire lock so nothing else can slip between the pairs of one burst.

Everything about the camera is optional. ``cv2`` and ``ultralytics`` are
imported lazily inside the worker threads, and any failure — missing packages,
a busy or absent device, a model that will not load — logs exactly ONE warning,
flips :attr:`CameraService.enabled` to ``False`` and leaves the voice pipeline
completely untouched.

Wiring (see :mod:`client.main`)::

    camera = CameraService(cfg.client.camera)
    camera.start(loop, ws.send_json, ws.send_bytes, send_lock=wire_lock)
    ...
    await camera.serve_request(request_id, burst)   # answers camera_request
    camera.stop()

``send_lock`` is the client's "one binary sequence at a time" lock: the server
routes incoming binary frames by the header that announced them, so a camera
JPEG must never slip between ``utterance_start`` and ``utterance_end`` or into
the middle of a screenshot. Presence pushes are simply skipped while that lock
is held (the user is talking — the server is not looking at faces anyway).
"""

from __future__ import annotations

import asyncio
import logging
import json
from copy import deepcopy
from pathlib import Path
import threading
import time
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from common.protocol import (
    CAMERA_BURST_MAX,
    CAMERA_FORMAT,
    CAMERA_REASON_PRESENCE,
    CAMERA_REASON_REQUEST,
    MSG_CAMERA_ERROR,
    MSG_CAMERA_FRAME,
    MSG_CAMERA_REQUEST,
    MSG_CAMERA_STATE,
)

log = logging.getLogger(__name__)

# --- tuning ---------------------------------------------------------------
#: Longest side of a pushed frame (SPEC v1.4: JPEG, largest side <= 1280).
MAX_SIDE_PX = 1280
#: JPEG quality of the pushed frames.
JPEG_QUALITY = 80
#: Quality for a ``full: true`` pull (find_object / SAM3): those frames skip the
#: downscale, so they also deserve far less compression noise.
FULL_JPEG_QUALITY = 95
#: Capture resolution requested from the device. OpenCV otherwise negotiates
#: the driver default, which on the C920 is 640x480 - SAM3 and the vision model
#: were being fed a blurry thumbnail of the room. The camera downgrades by
#: itself if it cannot do this, so asking is free.
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
#: A changed picture is announced at most this often (SPEC v1.4: >= 2 s apart).
STATE_DEBOUNCE_S = 2.0
#: Detections below this confidence are ignored when counting people/objects.
CONF_THRESHOLD = 0.5
#: Clamp for ``cfg.client.camera.fps``.
MIN_FPS = 0.2
MAX_FPS = 240.0
#: A frame older than this is not worth sending to the server any more.
STALE_FRAME_S = 10.0
#: How many consecutive failed ``VideoCapture.read()`` calls mean the device is
#: gone (a C920 briefly stumbles when another app grabs it).
MAX_READ_FAILURES = 30
#: How long :meth:`CameraService.stop` waits for its worker threads.
JOIN_TIMEOUT_S = 3.0
#: Burst extension (v1.4): how many frames one periodic presence push carries.
FACE_BURST = 3
#: Burst extension (v1.4): gap between frames of one burst, request or presence
#: ("~250 ms apart" per SPEC).
BURST_FRAME_INTERVAL_S = 0.25
#: Burst extension (v1.4): how long :meth:`CameraService._await_fresh_frame`
#: waits for a genuinely new capture-thread frame before giving up on one slot
#: of a burst (a stalled camera must not hang the whole request).
FRESH_FRAME_WAIT_S = 0.5

SendJson = Callable[[Dict[str, Any]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

__all__ = [
    "MSG_CAMERA_STATE",
    "MSG_CAMERA_FRAME",
    "MSG_CAMERA_REQUEST",
    "MSG_CAMERA_ERROR",
    "CAMERA_REASON_PRESENCE",
    "CAMERA_REASON_REQUEST",
    "MAX_SIDE_PX",
    "JPEG_QUALITY",
    "STATE_DEBOUNCE_S",
    "CONF_THRESHOLD",
    "FACE_BURST",
    "BURST_FRAME_INTERVAL_S",
    "CameraUnavailable",
    "CameraService",
]


class CameraUnavailable(RuntimeError):
    """The camera stack cannot be used (missing packages, no device, no model)."""


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a pydantic model / dataclass / mapping."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class CameraService:
    """Capture + YOLO worker pair that reports presence over the WebSocket.

    The object is always constructible: passing ``None`` (no ``camera`` section
    in the config) or a section with ``enabled: false`` yields a service whose
    :attr:`enabled` is ``False`` and whose :meth:`start` does nothing but keep
    the send callbacks, so an incoming ``camera_request`` still gets a proper
    ``camera_error`` instead of silence.
    """

    def __init__(self, cfg_camera: Any = None) -> None:
        import uuid
        self._track_epoch = uuid.uuid4().hex[:10]
        self._tracks = []
        self._last_tracks_at = 0.0
        self._enabled = bool(_attr(cfg_camera, "enabled", False))
        self.index = _as_int(_attr(cfg_camera, "index", 0), 0)
        self.stream_url = str(_attr(cfg_camera, 'stream_url', '') or '').strip()
        self.width = _as_int(_attr(cfg_camera, 'width', CAPTURE_WIDTH), CAPTURE_WIDTH)
        self.height = _as_int(_attr(cfg_camera, 'height', CAPTURE_HEIGHT), CAPTURE_HEIGHT)
        requested_fps = _as_float(_attr(cfg_camera, 'fps', 5), 5.)
        self.fps = 0. if requested_fps == 0 else min(MAX_FPS, max(MIN_FPS, requested_fps))
        self.half = bool(_attr(cfg_camera, 'half', True))
        self._recording_cfg = _attr(cfg_camera, 'frame_recording')
        self._frame_recorder = None
        self._archive_error = ''
        self._captured_count = self._inferred_count = 0
        self._metrics_at = time.monotonic()
        self._metrics_captured = self._metrics_inferred = 0
        self._presence_pending = threading.Event()
        self.model_name = str(_attr(cfg_camera, "model", "yolo11n.pt") or "yolo11n.pt")
        self.face_check_interval_s = max(
            0.5, _as_float(_attr(cfg_camera, "face_check_interval_s", 0.5), 0.5)
        )

        # --- wiring to the event loop ---
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send_json: Optional[SendJson] = None
        self._send_bytes: Optional[SendBytes] = None
        self._send_lock: Optional[asyncio.Lock] = None

        # --- threads ---
        self._stop_event = threading.Event()
        self._capture_ready = threading.Event()
        self._new_frame = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

        # --- latest frame ---
        self._frame_lock = threading.Lock()
        self._clip_lock = threading.Lock()
        self._frame: Any = None
        self._frame_ts = 0.0
        # One complete inference result, tied to the exact captured image.
        # Requests must never combine the latest capture with older tracks.
        self._detected_frame = None
        self._cv2: Any = None

        # --- reporting state ---
        self._warned = False
        self._detect_errors = 0
        self._send_errors = 0
        self._frame_seq = 0
        self._sent_state: Optional[Tuple[int, Tuple[Tuple[str, int], ...]]] = None
        self._sent_state_at = 0.0
        self._last_presence_push = 0.0

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """``False`` when the camera is off in the config or has failed.

        The voice pipeline never looks at anything else: a ``False`` here means
        "there is no camera on this machine", nothing more.
        """
        return self._enabled

    @property
    def running(self) -> bool:
        """True while the capture thread is alive and delivering frames."""
        thread = self._capture_thread
        return bool(self._enabled and thread is not None and thread.is_alive())

    def _fail(self, message: str) -> None:
        """Disable the camera permanently, logging exactly one warning."""
        self._enabled = False
        self._stop_event.set()
        if not self._warned:
            self._warned = True
            log.warning("Camera disabled: %s. The voice assistant keeps working.", message)
        else:  # pragma: no cover - a second failure after the first warning
            log.debug("Camera failure after it was already disabled: %s", message)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        send_json: SendJson,
        send_bytes: SendBytes,
        send_lock: Optional[asyncio.Lock] = None,
    ) -> bool:
        """Start the capture and detection threads.

        :param loop: the client's event loop; the worker threads schedule their
            sends onto it with ``run_coroutine_threadsafe``.
        :param send_json: coroutine function sending one JSON control frame.
        :param send_bytes: coroutine function sending one binary frame.
        :param send_lock: optional lock held around every header+binary pair so
            camera frames cannot interleave with streamed microphone audio.
        :returns: ``True`` if the threads were started.

        Returns immediately: importing ``ultralytics`` and opening the device
        take seconds and happen inside the workers, so the wake word is live
        while the camera is still warming up.
        """
        # Kept even when disabled: camera_request must still get an answer.
        self._loop = loop
        self._send_json = send_json
        self._send_bytes = send_bytes
        self._send_lock = send_lock

        if not self._enabled:
            log.info("Camera is disabled in the config - running voice only")
            return False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return True

        self._stop_event.clear()
        self._capture_ready.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="jarvis-camera-capture", daemon=True
        )
        self._infer_thread = threading.Thread(
            target=self._infer_loop, name="jarvis-camera-yolo", daemon=True
        )
        self._capture_thread.start()
        self._infer_thread.start()
        log.info(
            "Camera service starting: device index %d, %s at %.1f fps, "
            "one presence frame every %.1f s",
            self.index,
            self.model_name,
            self.fps,
            self.face_check_interval_s,
        )
        return True

    def stop(self) -> None:
        """Stop both threads and release the device (safe to call twice)."""
        self._stop_event.set()
        self._capture_ready.set()
        self._new_frame.set()
        for thread in (self._infer_thread, self._capture_thread):
            if thread is None or not thread.is_alive():
                continue
            thread.join(timeout=JOIN_TIMEOUT_S)
            if thread.is_alive():  # pragma: no cover - a stuck driver call
                log.debug("Camera thread %s did not stop in time", thread.name)
        if self._capture_thread is not None or self._infer_thread is not None:
            log.info("Camera service stopped")
        self._capture_thread = None
        self._infer_thread = None
        if self._frame_recorder is not None:
            self._frame_recorder.close()
            self._frame_recorder = None
        with self._frame_lock:
            self._frame = None
            self._frame_ts = 0.0
            self._detected_frame = None

    # ------------------------------------------------------------------
    # lazy imports
    # ------------------------------------------------------------------
    @staticmethod
    def _import_cv2() -> Any:
        # Not just ImportError: a half-installed OpenCV raises DLL/numpy errors.
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise CameraUnavailable(
                "OpenCV is not available (pip install -r client/requirements-camera.txt): "
                f"{exc}"
            ) from exc
        return cv2

    @staticmethod
    def _import_yolo() -> Any:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # noqa: BLE001 - torch/ultralytics import errors
            raise CameraUnavailable(
                "Ultralytics is not available (pip install -r client/requirements-camera.txt): "
                f"{exc}"
            ) from exc
        return YOLO

    # ------------------------------------------------------------------
    # capture thread
    # ------------------------------------------------------------------
    def _request_resolution(self, cv2: Any, capture: Any) -> None:
        """Ask the device for :data:`CAPTURE_WIDTH` x :data:`CAPTURE_HEIGHT`.

        Never fatal: a camera that cannot do it simply keeps its own mode, and
        the negotiated size is logged so a blurry frame is obvious in the log.
        """
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            log.info("Camera %d capturing at %dx%d", self.index, width, height)
            if hasattr(cv2, 'CAP_PROP_FPS'):
                log.info('Camera negotiated capture rate: %.1f FPS', capture.get(cv2.CAP_PROP_FPS) or 0)
        except Exception as exc:  # noqa: BLE001 - resolution is best-effort
            log.debug("Could not set the capture resolution: %s", exc)

    def _open_capture(self, cv2: Any) -> Any:
        """Open ``cfg.camera.index``, retrying with DirectShow on Windows."""
        if self.stream_url:
            from urllib.parse import urlsplit
            parsed = urlsplit(self.stream_url)
            if parsed.scheme not in ('rtsp', 'rtsps') or not parsed.hostname:
                raise CameraUnavailable('camera.stream_url must be an RTSP URL')
            # FFmpeg reads native resolution. Timeouts bound reconnect/shutdown;
            # the capture thread continuously drains the stream to limit delay.
            try:
                capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2500,
                ])
            except Exception:
                raise CameraUnavailable('Could not open the RTSP camera stream') from None
            if capture is not None and capture.isOpened():
                log.info('Network camera connected (RTSP, native resolution)')
                return capture
            if capture is not None:
                capture.release()
            raise CameraUnavailable('RTSP camera unavailable; check its local address and camera account')
        capture = cv2.VideoCapture(self.index)
        if capture is not None and capture.isOpened():
            self._request_resolution(cv2, capture)
            return capture
        if capture is not None:
            try:
                capture.release()
            except Exception:  # pragma: no cover - defensive
                pass
        # MSMF (the Windows default) refuses some UVC webcams that DirectShow
        # opens without complaint, so the fallback is worth the extra attempt.
        backend = getattr(cv2, "CAP_DSHOW", None)
        if backend is not None:
            capture = cv2.VideoCapture(self.index, backend)
            if capture is not None and capture.isOpened():
                log.debug("Camera %d opened through the DirectShow backend", self.index)
                self._request_resolution(cv2, capture)
                return capture
            if capture is not None:
                try:
                    capture.release()
                except Exception:  # pragma: no cover - defensive
                    pass
        raise CameraUnavailable(
            f"camera index {self.index} could not be opened (is it unplugged or in use?)"
        )

    def _capture_loop(self) -> None:
        capture = None
        cv2 = None
        try:
            try:
                cv2 = self._import_cv2()
                capture = self._open_capture(cv2)
                self._cv2 = cv2
            except CameraUnavailable as exc:
                if not self.stream_url or cv2 is None:
                    self._fail(str(exc))
                    return
                log.warning('Network camera unavailable; waiting for it to reconnect')
                while not self._stop_event.wait(2):
                    try:
                        capture = self._open_capture(cv2)
                        self._cv2 = cv2
                        break
                    except CameraUnavailable:
                        continue
                if self._stop_event.is_set():
                    return
            except Exception as exc:  # noqa: BLE001 - never let a driver kill the client
                self._fail(f"could not start the camera: {exc}")
                return
            finally:
                self._capture_ready.set()

            log.info("Camera %d opened", self.index)
            failures = 0
            while not self._stop_event.is_set():
                try:
                    ok, frame = capture.read()
                except Exception as exc:  # noqa: BLE001 - driver hiccup
                    ok, frame = False, None
                    log.debug("Camera read error: %s", exc)
                if not ok or frame is None:
                    failures += 1
                    if failures >= (1 if self.stream_url else MAX_READ_FAILURES):
                        if self.stream_url:
                            capture.release()
                            with self._frame_lock:
                                self._frame = None
                                self._frame_ts = 0.0
                                self._detected_frame = None
                            log.warning('Network camera disconnected; retrying in 2 seconds')
                            while not self._stop_event.wait(2):
                                try:
                                    capture = self._open_capture(cv2)
                                    failures = 0
                                    break
                                except CameraUnavailable:
                                    continue
                            continue
                        self._fail("the camera stopped delivering frames")
                        return
                    self._stop_event.wait(0.2)
                    continue
                failures = 0
                with self._frame_lock:
                    self._frame = frame
                    self._frame_ts = time.monotonic()
                    self._captured_count += 1
                # read() waits for the device; extra sleeps reduce capture FPS.
                self._new_frame.set()
        finally:
            self._capture_ready.set()
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:  # pragma: no cover - teardown
                    log.debug("Error while releasing the camera: %s", exc)

    def _latest_frame(self) -> Tuple[Any, float]:
        """The newest captured frame and its age in seconds (``None``, ``inf``)."""
        with self._frame_lock:
            frame = self._frame
            ts = self._frame_ts
        if frame is None:
            return None, float("inf")
        return frame, max(0.0, time.monotonic() - ts)

    def _latest_frame_ts(self) -> Tuple[Any, float]:
        """The newest captured frame and its OWN monotonic timestamp (not age).

        Used by :meth:`_await_fresh_frame` to tell burst frames apart: an age
        alone cannot say whether two reads a few milliseconds apart landed on
        the same capture-thread frame or two different ones.
        """
        with self._frame_lock:
            return self._frame, self._frame_ts

    def _await_fresh_frame(self, newer_than: float) -> Tuple[Any, float]:
        """Block briefly for a capture-thread frame newer than ``newer_than``.

        One slot of a burst must never resend the same JPEG twice: the capture
        thread normally refreshes far faster than :data:`BURST_FRAME_INTERVAL_S`,
        but this polls a little longer (up to :data:`FRESH_FRAME_WAIT_S`) as a
        safety net for a slow or throttled camera, rather than trusting the
        sleep alone. Gives up and returns ``(None, 0.0)`` when nothing fresh (or
        nothing at all, or nothing fresher than :data:`STALE_FRAME_S`) turns up
        in time -- never blocks longer than that, never raises.
        """
        deadline = time.monotonic() + FRESH_FRAME_WAIT_S
        while True:
            frame, ts = self._latest_frame_ts()
            if frame is not None and ts > newer_than:
                age = max(0.0, time.monotonic() - ts)
                if age <= STALE_FRAME_S:
                    return frame, ts
            if time.monotonic() >= deadline:
                return None, 0.0
            time.sleep(0.02)

    def _cache_detection(self, frame: Any, captured_at: float) -> None:
        """Publish a same-frame snapshot from the sole YOLO worker."""
        with self._frame_lock:
            self._detected_frame = (frame, captured_at, deepcopy(self._tracks))

    def _await_detected_frame(self, newer_than: float) -> tuple:
        """Wait briefly for normal inference; never run extra YOLO work.

        Runs in the request's worker thread. If inference is unavailable or
        slower than the request budget, the caller can still send a fresh
        capture with unknown tracks instead of borrowing stale detections.
        """
        deadline = time.monotonic() + FRESH_FRAME_WAIT_S
        while not self._stop_event.is_set():
            with self._frame_lock:
                detected = self._detected_frame
            if detected is not None:
                frame, captured_at, tracks = detected
                if captured_at > newer_than and time.monotonic() - captured_at <= STALE_FRAME_S:
                    return frame, captured_at, tracks
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._stop_event.wait(min(.02, remaining))
        return None, 0.0, None

    def _capture_burst_sync(self, count: int, full: bool = False, include_tracks: bool = False) -> list:
        """Capture up to ``count`` fresh frames ~250 ms apart, JPEG-encoded.

        Synchronous by design: called either via ``asyncio.to_thread`` (a
        ``camera_request``, which has no frame in hand yet) or directly from
        the YOLO worker thread (a presence push, which is already off the
        event loop). Stops early and returns whatever was captured so far the
        moment a fresh frame cannot be had or its encoding fails -- a partial
        burst is still useful to the caller (the server's ``best_face`` picks
        the best of whatever arrives), this never raises.

        :param full: v1.6 -- skip the usual :data:`MAX_SIDE_PX` downscale for
            every frame of this burst (``find_object`` wants native
            resolution); ``False`` (the default) is the pre-v1.6 behaviour.
        :param include_tracks: prefer a newly captured frame processed by the
            normal inference worker, with tracks from that exact image.
        :returns: ``[(jpeg_bytes, width, height), …]``, shortest at ``[]``;
            ``include_tracks`` adds a fourth item (tracks or ``None``).
        """
        pairs: list = []
        newer_than = time.monotonic() if include_tracks else -1.0
        for seq in range(1, max(1, count) + 1):
            if seq > 1:
                time.sleep(BURST_FRAME_INTERVAL_S)
            tracks = None
            frame, ts = None, 0.0
            if include_tracks:
                frame, ts, tracks = self._await_detected_frame(newer_than)
            if frame is None:
                frame, ts = self._await_fresh_frame(newer_than)
            if frame is None:
                break
            newer_than = ts
            try:
                jpeg, width, height = self._encode(frame, full=full)
            except Exception as exc:  # noqa: BLE001 - encoding must not kill the caller
                log.debug("Could not encode burst frame %d/%d: %s", seq, count, exc)
                break
            pair = (jpeg, width, height)
            pairs.append((*pair, tracks) if include_tracks else pair)
        return pairs

    # ------------------------------------------------------------------
    # detection thread
    # ------------------------------------------------------------------
    def _infer_loop(self) -> None:
        self._capture_ready.wait()
        if self._stop_event.is_set() or not self._enabled:
            return
        try:
            yolo_class = self._import_yolo()
            model = yolo_class(self.model_name)
        except CameraUnavailable as exc:
            self._fail(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - weights download / CUDA errors
            self._fail(f"the YOLO model {self.model_name!r} could not be loaded: {exc}")
            return
        log.info('YOLO model %s loaded, CUDA FP16=%s, FPS limit=%s',
                 self.model_name, self.half, self.fps or 'unlimited')
        if _attr(self._recording_cfg, 'enabled', False):
            try:
                from client.frame_recording import FrameRecorder
                self._frame_recorder = FrameRecorder(self._recording_cfg, self._cv2)
            except Exception as exc:
                self._archive_error = type(exc).__name__
                log.exception('Camera recording could not start')

        interval = 1.0 / self.fps if self.fps else 0.
        last_frame_ts = 0.
        while not self._stop_event.is_set():
            started = time.monotonic()
            self._new_frame.clear()
            frame, frame_ts = self._latest_frame_ts()
            if frame is None or frame_ts <= last_frame_ts:
                self._new_frame.wait(.1)
                continue
            last_frame_ts = frame_ts
            age = started - frame_ts
            if frame is not None and age <= STALE_FRAME_S:
                try:
                    persons, objects = self._detect(model, frame)
                except Exception as exc:  # noqa: BLE001 - inference must not kill us
                    self._detect_errors += 1
                    if self._detect_errors == 1:
                        log.warning("YOLO inference failed: %s", exc)
                    else:
                        log.debug("YOLO inference failed (%d): %s", self._detect_errors, exc)
                    if self._detect_errors >= 10:
                        self._fail("YOLO inference keeps failing")
                        return
                else:
                    self._detect_errors = 0
                    self._inferred_count += 1
                    self._cache_detection(frame, frame_ts)
                    self._publish_state(persons, objects)
                    if persons >= 1:
                        if self._frame_recorder is not None:
                            self._frame_recorder.submit(frame, time.time() - (time.monotonic() - frame_ts),
                                {'persons': persons, 'tracks': list(self._tracks), 'objects': objects,
                                 'camera_id': 'network' if self.stream_url else f'usb:{self.index}',
                                 'model': self.model_name}, stop_event=self._stop_event)
                        self._maybe_push_presence(frame)
                    self._report_performance()
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _detect(self, model: Any, frame: Any) -> Tuple[int, Dict[str, int]]:
        """Count people and objects in one frame by YOLO class name."""
        from pathlib import Path
        results = model.track(
            source=frame,
            persist=True,
            tracker=str(Path(__file__).with_name('room-tracker.yaml')),
            conf=CONF_THRESHOLD,
            verbose=False,
            device=0,
            imgsz=640,
            half=self.half,
        )
        counts: Dict[str, int] = {}
        tracks = []
        for result in results or []:
            names = getattr(result, "names", None) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            classes = getattr(boxes, "cls", None)
            confidences = getattr(boxes, "conf", None)
            if classes is None or confidences is None:
                continue
            class_list = classes.tolist() if hasattr(classes, "tolist") else list(classes)
            conf_list = (
                confidences.tolist() if hasattr(confidences, "tolist") else list(confidences)
            )
            ids = boxes.id.tolist() if getattr(boxes, 'id', None) is not None else []
            positions = boxes.xyxyn.tolist() if getattr(boxes, 'xyxyn', None) is not None else []
            for box_index, (class_id, confidence) in enumerate(zip(class_list, conf_list)):
                if float(confidence) < CONF_THRESHOLD:
                    continue
                index = int(class_id)
                label = str(names.get(index, index) if isinstance(names, dict) else index)
                counts[label] = counts.get(label, 0) + 1
                if label == 'person' and box_index < len(ids):
                    tracks.append({'id': f'{self._track_epoch}:{int(ids[box_index])}',
                                   'box': [max(0., min(1., float(v))) for v in positions[box_index]]})
        self._tracks = tracks
        persons = counts.pop("person", 0)
        return int(persons), counts

    def _report_performance(self):
        now = time.monotonic()
        elapsed = now - self._metrics_at
        if elapsed < 5:
            return
        stats = {'ts': time.time(), 'model': self.model_name, 'half': self.half,
                 'fps_limit': self.fps,
                 'capture_fps': round((self._captured_count - self._metrics_captured) / elapsed, 2),
                 'yolo_fps': round((self._inferred_count - self._metrics_inferred) / elapsed, 2),
                 'captured_frames': self._captured_count, 'processed_frames': self._inferred_count,
                 'recording': self._frame_recorder.stats() if self._frame_recorder else None,
                 'archive_error': self._archive_error}
        self._metrics_at, self._metrics_captured, self._metrics_inferred = now, self._captured_count, self._inferred_count
        log.info('Camera performance: %s', stats)
        try:
            path = Path(__file__).resolve().parents[1] / 'data/camera-performance.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix('.pending')
            pending.write_text(json.dumps(stats), encoding='utf-8')
            pending.replace(path)
        except OSError:
            log.debug('Could not save camera performance counters', exc_info=True)

    # ------------------------------------------------------------------
    # camera_state (SPEC v1.4)
    # ------------------------------------------------------------------
    def _publish_state(self, persons: int, objects: Dict[str, int]) -> None:
        """Send ``camera_state`` when the picture changed, at most every 2 s.

        YOLO flickers object counts (a bottle drifting between 1 and 2 across
        frames), which used to re-announce the "change" every debounce window.
        A PERSON-count change is reported at once; an object-only change must
        stay identical for two consecutive checks before it is sent.
        """
        # Change detection uses the person count and the SET of labels only:
        # YOLO endlessly flickers object counts (a bottle drifting between 1
        # and 2), and re-announcing every count change spammed the server every
        # debounce window. Counts still ride along in the payload when a real
        # change (new/removed label, person count) is announced.
        key = (int(persons), tuple(sorted(objects.keys())))
        now = time.monotonic()
        heartbeat = now - self._last_tracks_at >= .4
        if key == self._sent_state and not heartbeat:
            return
        if now - self._sent_state_at < STATE_DEBOUNCE_S and not heartbeat:
            # Too soon after the last report: the state is recomputed every tick,
            # so the change is simply announced by one of the next ones. Short
            # flickers (a misdetected chair) never reach the server at all.
            return
        self._sent_state = key
        self._sent_state_at = now
        self._last_tracks_at = now
        payload = {
            "type": MSG_CAMERA_STATE,
            "persons": int(persons),
            "tracks": list(self._tracks),
            "objects": {label: int(count) for label, count in sorted(objects.items())},
        }
        # Person-count changes are worth a console line; object-label churn
        # (a phone appearing/disappearing) only spams it - keep that at DEBUG.
        persons_changed = getattr(self, "_last_logged_persons", None) != int(persons)
        self._last_logged_persons = int(persons)
        log.log(
            logging.INFO if persons_changed else logging.DEBUG,
            "Camera: %d person(s), objects: %s",
            persons,
            ", ".join(f"{label} x{count}" for label, count in sorted(objects.items())) or "none",
        )
        self._submit(self._send_state(payload))

    async def _send_state(self, payload: Dict[str, Any]) -> None:
        send_json = self._send_json
        if send_json is None:
            return
        try:
            await send_json(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead socket is normal here
            self._note_send_failure("camera_state", exc)

    # ------------------------------------------------------------------
    # camera_frame (SPEC v1.4)
    # ------------------------------------------------------------------
    def _maybe_push_presence(self, frame=None) -> None:
        """Push a :data:`FACE_BURST`-frame burst while somebody is visible (v1.4 burst).

        JPEG encoding runs off the YOLO thread. At most one presence update is
        pending; its track boxes are captured together with its source frame.
        """
        now = time.monotonic()
        if self._presence_pending.is_set() or now - self._last_presence_push < self.face_check_interval_s:
            return
        lock = self._send_lock
        if lock is not None and lock.locked():
            # The microphone is streaming an utterance (or another burst is
            # going out): the server would read our JPEGs as audio. Skip the
            # whole push -- including the capture -- the next one comes in
            # face_check_interval_s.
            log.debug("Skipping a presence burst - the socket is busy")
            return
        self._last_presence_push = now
        self._frame_seq += 1
        frame_id = f'p{self._frame_seq}'
        self._presence_pending.set()
        if self._submit(self._encode_and_push_presence(frame, frame_id, list(self._tracks))) is None:
            self._presence_pending.clear()

    async def _encode_and_push_presence(self, frame, frame_id, tracks):
        try:
            # full=True: face recognition needs a CRISP face. The old 1280px /
            # quality-80 presence frame made the owner's own face score around
            # the match threshold; a native-resolution frame fixes that at the
            # source instead of lowering the bar.
            pairs = ([await asyncio.to_thread(self._encode, frame, full=True)] if frame is not None
                     else await asyncio.to_thread(self._capture_burst_sync, 1, full=True))
            if pairs:
                await self._send_burst(frame_id, CAMERA_REASON_PRESENCE, pairs, skip_if_busy=True, tracks=tracks)
        except Exception as exc:  # noqa: BLE001 - capture/encoding must not kill the thread
            log.debug("Could not capture the presence burst: %s", exc)
            return
        finally:
            self._presence_pending.clear()

    def _encode(self, frame: Any, full: bool = False) -> Tuple[bytes, int, int]:
        """Downscale to :data:`MAX_SIDE_PX` and encode as JPEG q80.

        :param full: v1.6 -- skip the downscale entirely for this frame
            (``find_object`` wants the detector to see native resolution).
        """
        cv2 = self._cv2
        if cv2 is None:  # pragma: no cover - only reachable before the first frame
            raise CameraUnavailable("OpenCV is not loaded")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if width <= 0 or height <= 0:
            raise CameraUnavailable("the camera returned an empty frame")
        longest = max(width, height)
        if not full and longest > MAX_SIDE_PX:
            scale = MAX_SIDE_PX / float(longest)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        quality = FULL_JPEG_QUALITY if full else JPEG_QUALITY
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise CameraUnavailable("JPEG encoding failed")
        return bytes(buffer.tobytes() if hasattr(buffer, "tobytes") else buffer), width, height

    async def _send_burst(
        self,
        frame_id: str,
        reason: str,
        pairs: list,
        skip_if_busy: bool = False,
        tracks=None,
    ) -> None:
        """Send ``pairs`` as that many ``camera_frame`` header+binary pairs.

        All of them go out under ONE hold of the wire lock (SPEC v1.4 burst),
        so nothing else -- streamed microphone audio, a screenshot -- can slip
        between the frames of a single burst. ``skip_if_busy`` mirrors the old
        single-frame behaviour for presence pushes: if the wire is busy right
        now, drop the whole burst rather than wait for it (the next one comes
        in ``face_check_interval_s``); a server-requested burst always waits.
        """
        send_json, send_bytes = self._send_json, self._send_bytes
        if send_json is None or send_bytes is None or not pairs:
            return
        lock = self._send_lock
        if lock is not None and lock.locked() and skip_if_busy:
            log.debug("Skipping a %s burst %s - the socket is busy", reason, frame_id)
            return
        total = len(pairs)
        log.debug(
            "Sending a %s frame burst %s: %d frame(s)", reason, frame_id, total
        )
        try:
            if lock is None:
                for seq, pair in enumerate(pairs, start=1):
                    jpeg, width, height = pair[:3]
                    frame_tracks = pair[3] if len(pair) > 3 else tracks
                    await self._send_pair(
                        send_json, send_bytes, frame_id, reason, seq, total, jpeg, width, height, frame_tracks
                    )
            else:
                async with lock:
                    for seq, pair in enumerate(pairs, start=1):
                        jpeg, width, height = pair[:3]
                        frame_tracks = pair[3] if len(pair) > 3 else tracks
                        await self._send_pair(
                            send_json, send_bytes, frame_id, reason, seq, total, jpeg, width, height, frame_tracks
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconnects are routine
            self._note_send_failure("camera_frame", exc)

    @staticmethod
    async def _send_pair(
        send_json: SendJson,
        send_bytes: SendBytes,
        frame_id: str,
        reason: str,
        seq: int,
        total: int,
        jpeg: bytes,
        width: int,
        height: int,
        tracks=None,
    ) -> None:
        """Send one ``camera_frame`` header plus its single binary frame."""
        header = {
            "type": MSG_CAMERA_FRAME,
            "id": frame_id,
            "reason": reason,
            "format": CAMERA_FORMAT,
            "w": int(width),
            "h": int(height),
            "seq": int(seq),
            "of": int(total),
            "tracks": tracks,
        }
        log.debug(
            "Sending camera frame %s (%s) %d/%d: %dx%d, %d bytes",
            frame_id, reason, seq, total, width, height, len(jpeg),
        )
        await send_json(header)
        await send_bytes(jpeg)

    async def _send_error(self, request_id: str, error: str) -> None:
        """Mirror of ``screenshot_error`` (SPEC v1.4): no binary frame follows."""
        send_json = self._send_json
        if send_json is None:
            return
        try:
            await send_json(
                {"type": MSG_CAMERA_ERROR, "id": str(request_id), "error": str(error)}
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._note_send_failure("camera_error", exc)

    def _note_send_failure(self, kind: str, exc: BaseException) -> None:
        """A failed send means the socket is down; re-announce state later."""
        self._sent_state = None  # force a fresh camera_state after the reconnect
        self._send_errors += 1
        if self._send_errors == 1:
            log.info("Could not send %s (%s) - retrying once reconnected", kind, exc)
        else:
            log.debug("Could not send %s (%d): %s", kind, self._send_errors, exc)

    # ------------------------------------------------------------------
    # camera_request (SPEC v1.4)
    # ------------------------------------------------------------------
    async def serve_request(self, request_id: str, burst: int = 1, full: bool = False) -> None:
        """Answer the server's ``camera_request`` with ``burst`` frame(s) (v1.4 burst).

        Captures ``burst`` frames roughly :data:`BURST_FRAME_INTERVAL_S` apart,
        each one confirmed fresh from the capture thread (never the same JPEG
        resent twice), then sends them as that many header+binary pairs sharing
        ``request_id`` -- all under one hold of the wire lock -- each header
        carrying ``seq``/``of``. ``burst=1`` (the default, and every pre-burst
        caller) behaves exactly as the single-frame request always did.
        Each frame carries its own YOLO tracks when normal inference delivers
        a fresh result in time; otherwise tracks are unknown (``None``).

        :param full: v1.6 -- honor the request's ``"full": true`` (skip the
            usual downscale for every frame of this pull; ``find_object``
            wants native resolution).

        Never raises: a missing camera, a stale frame or an encoding error all
        come back to the server as ``camera_error`` so its tool call fails fast
        instead of waiting for a timeout; a burst that manages SOME frames but
        not all of them still answers with what it has (a partial burst beats
        none for the server's best-face selection).
        """
        request_id = str(request_id or "")
        if not self._enabled:
            await self._send_error(request_id, "the camera is not available on this client")
            return
        count = max(1, min(_as_int(burst, 1), CAMERA_BURST_MAX))

        try:
            pairs = await asyncio.to_thread(self._capture_burst_sync, count, full, include_tracks=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - capture/encoding must not kill the client
            error = str(exc).strip() or exc.__class__.__name__
            log.warning("Could not capture the requested camera frame(s): %s", error)
            await self._send_error(request_id, error)
            return

        if not pairs:
            reason = (
                "the camera has no frame yet (still starting up)"
                if self.running
                else "the camera is not running"
            )
            await self._send_error(request_id, reason)
            return

        log.info(
            "Answering the camera request (id=%s): %d/%d frame(s)",
            request_id or "?", len(pairs), count,
        )
        await self._send_burst(request_id, CAMERA_REASON_REQUEST, pairs)

    # ------------------------------------------------------------------
    # thread -> event loop
    # ------------------------------------------------------------------
    def _submit(self, coro: Awaitable[None]) -> Any:
        """Run a coroutine on the client's loop from a worker thread."""
        loop = self._loop
        if loop is None or loop.is_closed() or self._stop_event.is_set():
            close = getattr(coro, "close", None)
            if callable(close):  # never leave a coroutine un-awaited
                close()
            return
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - the loop may be shutting down
            log.debug("Could not hand the camera message to the event loop: %s", exc)
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            return
        future.add_done_callback(self._drain_future)
        return future

    @staticmethod
    def _drain_future(future: Any) -> None:
        """Consume the result so a failed send is never an unretrieved exception."""
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001 - already logged where it happened
            log.debug("Camera send task ended with: %s", exc)
