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
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from client.body_crops import CropSchedule, encode_crop
from client.privacy import PrivacyMode
from client.tracking import TrackRegistry, write_tracker_config
from common.attention_objects import attention_group
from common.frame_zones import (
    FrameZone,
    mask_polygons,
    masked_at,
    masks_rev,
    parse_zones,
    zone_at,
    zones_rev,
)
from common.ids import new_ulid
from common.protocol import (
    CAMERA_BURST_MAX,
    CAMERA_FORMAT,
    CAMERA_REASON_PRESENCE,
    CAMERA_REASON_REQUEST,
    MSG_BODY_CROP,
    MSG_CAMERA_ERROR,
    MSG_CAMERA_FRAME,
    MSG_CAMERA_REQUEST,
    MSG_CAMERA_STATE,
    MSG_OBJECT_EVENT,
    MSG_ROOM_HEALTH,
    MSG_TRACKS,
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
#: While somebody is in the room - and this long after the last sighting -
#: the detector runs frame by frame instead of at the configured FPS cap. The
#: cap is a budget for an empty room; a fast pass-by is exactly the moment the
#: next frames decide whether the person is seen at all (ТЗ F-201).
ACTIVE_DETECTION_HOLD_S = 3.0
#: How many consecutive failed ``VideoCapture.read()`` calls mean the device is
#: gone (a C920 briefly stumbles when another app grabs it).
MAX_READ_FAILURES = 30
#: Reopening a local camera after that: first pause, and the ceiling the pause
#: grows to. Владелец 2026-09-23: комната должна возвращаться сама, когда
#: устройство освободится, а не ждать перезапуска клиента.
CAMERA_RETRY_MIN_S = 2.0
CAMERA_RETRY_MAX_S = 30.0
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
#: ТЗ F-201: a quick pass-by is shorter than the heavy detector's own frame.
#: YOLO11x needs ~300 ms per frame, so on the owner's PC the accurate detector
#: samples the room about three times a second and a person crossing it in half
#: a second can be gone before the next sample. The light guard runs between
#: those frames and turns the first sighting into the presence burst.
QUICK_PASS_MODEL = "yolo11n.pt"
#: Default guard rate (``cfg.client.camera.quick_fps``).
QUICK_PASS_FPS = 8.0
#: Guard confidence. Lower than :data:`CONF_THRESHOLD` on purpose: the guard
#: only has to notice that somebody is there, and a missed pass-by costs the
#: room its notification. The heavy detector decides who it was.
QUICK_PASS_CONF = 0.35
#: The guard infers at the same size the heavy detector uses, so the two agree
#: about what "a person" means at this camera's distance.
QUICK_PASS_IMGSZ = 640
#: Two guard-triggered bursts closer than this are the same person walking
#: through, not two people: the hub needs a burst, not a stream of them.
QUICK_PASS_COOLDOWN_S = 4.0
#: Владелец 2026-09-24: «если быстро перед камерой пройти то почему-то не
#: записывает человека». Клип начинался в момент запроса, а правило
#: срабатывает на кадр-два позже — человек к этому времени уже вышел, и в
#: видео пустая комната. Поэтому поток захвата держит последние секунды
#: маленьких JPEG (без рамок: рамки и имена рисуются при записи, когда хаб уже
#: прислал, кого он узнал), и клип начинается с них.
PREROLL_SECONDS = 4.0
PREROLL_FPS = 8.0
#: Side a pre-roll frame is scaled to before encoding; the same cap the clip
#: writer scales live frames to, so the video keeps one constant size.
PREROLL_MAX_SIDE = 960
#: Владелец 2026-09-24: «открыть камеру и чтобы оно показывало видео с камеры на
#: экране и все детекции». The live preview draws the boxes of the LAST inference
#: straight from here: every class the detector saw (not only people and the
#: attention groups), with its confidence. Only the newest frame is kept, so this
#: costs one small list per inference and never touches the wire.
LIVE_PREVIEW_MAX_DETECTIONS = 40

SendJson = Callable[[dict[str, Any]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

__all__ = [
    "MSG_CAMERA_STATE",
    "MSG_OBJECT_EVENT",
    "MSG_TRACKS",
    "MSG_BODY_CROP",
    "MSG_CAMERA_FRAME",
    "MSG_CAMERA_REQUEST",
    "MSG_CAMERA_ERROR",
    "MSG_ROOM_HEALTH",
    "CAMERA_REASON_PRESENCE",
    "CAMERA_REASON_REQUEST",
    "MAX_SIDE_PX",
    "JPEG_QUALITY",
    "STATE_DEBOUNCE_S",
    "CONF_THRESHOLD",
    "FACE_BURST",
    "BURST_FRAME_INTERVAL_S",
    "precision_kwargs",
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


@lru_cache(maxsize=1)
def _fp16_arg_name() -> str:
    """The keyword the installed Ultralytics wants for FP16 inference.

    Ultralytics 8.4 replaced ``half`` with ``quantize`` (16 = FP16, ``None`` =
    FP32). The old name still works, but it prints a deprecation line for EVERY
    prediction - the room PC's console filled with "WARNING 'half' is deprecated
    and will be removed in the future. Use 'quantize' instead." and nothing else
    could be read on it. Older builds only understand ``half``, so the name is
    taken from the build that is actually installed.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception as exc:  # noqa: BLE001 - no ultralytics: keep the old name
        log.debug("Ultralytics precision argument unknown (%s); using 'half'", exc)
        return "half"
    return "quantize" if "quantize" in DEFAULT_CFG_DICT else "half"


def precision_kwargs(half: bool) -> dict[str, Any]:
    """The FP16 switch as ``{"quantize": 16}`` or ``{"half": True}`` (see F-201).

    ``half=False`` must clear the precision instead of inheriting a higher one,
    and the new name spells that as ``quantize=None`` rather than ``False``.
    """
    if _fp16_arg_name() == "quantize":
        return {"quantize": 16 if half else None}
    return {"half": bool(half)}


def yolo_placement() -> tuple[Any, bool]:
    """Where YOLO runs: GPU 0 with FP16 when torch can, otherwise the CPU.

    A room PC without CUDA - a friend's laptop, for instance - answered every
    ``device=0`` with "Invalid CUDA 'device=0' requested", and the camera stack
    then switched itself off ("YOLO inference keeps failing") even though CPU
    inference handles a single 1080p stream. Detecting the card here keeps one
    inference stream working instead of losing the camera entirely.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch is optional for the client
        log.debug("torch is unavailable (%s); YOLO will run on the CPU", exc)
        return "cpu", False
    try:
        if bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0:
            return 0, True
    except Exception as exc:  # noqa: BLE001 - a broken CUDA init means the CPU
        log.debug("CUDA cannot be used (%s); YOLO will run on the CPU", exc)
    return "cpu", False


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
        #: ТЗ F-201: the ids of the people in the frame, with a thirty-second
        #: memory so a person who steps out and comes back keeps the same id.
        self.tracks = TrackRegistry()
        self._track_reports: list = []
        #: Владелец 2026-09-24: the boxes of the last inference for the live
        #: preview window (see :data:`LIVE_PREVIEW_MAX_DETECTIONS`).
        self._last_detections: list[dict[str, Any]] = []
        self._sent_track_ids: tuple[str, ...] = ()
        self._sent_tracks_at = 0.0
        self._tracks_message = bool(_attr(cfg_camera, 'tracks_message', True))
        #: Владелец 2026-09-24: the last seconds of frames, so an alert clip of
        #: somebody who walked past quickly still shows that person (see
        #: :data:`PREROLL_SECONDS`). Frames are small JPEGs; the boxes and the
        #: names are drawn when the clip is written.
        self._preroll: deque = deque(maxlen=max(2, int(PREROLL_SECONDS * PREROLL_FPS) + 2))
        self._preroll_lock = threading.Lock()
        self._preroll_at = 0.0
        #: ТЗ F-202: on appearance, every 2 s, and whenever the view changes.
        self._crop_schedule = CropSchedule()
        self._enabled = bool(_attr(cfg_camera, "enabled", False))
        self.index = _as_int(_attr(cfg_camera, "index", 0), 0)
        self.stream_url = str(_attr(cfg_camera, 'stream_url', '') or '').strip()
        self.width = _as_int(_attr(cfg_camera, 'width', CAPTURE_WIDTH), CAPTURE_WIDTH)
        self.height = _as_int(_attr(cfg_camera, 'height', CAPTURE_HEIGHT), CAPTURE_HEIGHT)
        requested_fps = _as_float(_attr(cfg_camera, 'fps', 5), 5.)
        self.fps = 0. if requested_fps == 0 else min(MAX_FPS, max(MIN_FPS, requested_fps))
        #: ТЗ F-312 works on whatever this PC has: GPU 0 when the card is
        #: usable, the CPU otherwise. FP16 only ever means something on a GPU.
        self.device, gpu_ready = yolo_placement()
        self.half = bool(_attr(cfg_camera, 'half', True)) and gpu_ready
        self._recording_cfg = _attr(cfg_camera, 'frame_recording')
        self._frame_recorder = None
        self._archive_error = ''
        self._captured_count = self._inferred_count = 0
        self._metrics_at = time.monotonic()
        self._metrics_captured = self._metrics_inferred = 0
        self._presence_pending = threading.Event()
        #: ТЗ F-303: пока privacy-режим включён, кадры не уходят вообще —
        #: проверяется в каждой точке, где клиент отдаёт картинку.
        self._privacy_mode = PrivacyMode(str(_attr(cfg_camera, 'language', 'en') or 'en'))
        #: ТЗ F-309: зоны кадра дома. Маска закрашивается ЗДЕСЬ, до JPEG,
        #: поэтому кадр уходит в хаб уже без неё; хаб сверяет отпечаток масок
        #: (``zones_rev`` в заголовке) со своим конфигом.
        self._zones: list[FrameZone] = parse_zones(_attr(cfg_camera, 'zones', None))
        self.model_name = str(_attr(cfg_camera, "model", "yolo11n.pt") or "yolo11n.pt")
        #: ТЗ F-201: the light guard (``_quick_gate_loop``) that watches for the
        #: first sign of a person between the heavy detector's frames. An empty
        #: name, or the same weights as the heavy model, turns it off.
        self.quick_model_name = str(_attr(cfg_camera, "quick_model", QUICK_PASS_MODEL) or "").strip()
        self.quick_fps = max(
            0.0, _as_float(_attr(cfg_camera, "quick_fps", QUICK_PASS_FPS), QUICK_PASS_FPS)
        )
        self.face_check_interval_s = max(
            0.5, _as_float(_attr(cfg_camera, "face_check_interval_s", 0.5), 0.5)
        )
        #: ТЗ F-312: профиль детекции выбирается измеренной задержкой при
        #: старте. Пока выбор выключен, работают ровно значения конфига выше.
        self._camera_cfg = cfg_camera
        self.auto_profile = bool(_attr(cfg_camera, "auto_profile", False))
        self.profile_measure_frames = max(
            1, _as_int(_attr(cfg_camera, "profile_measure_frames", 3), 3)
        )
        #: Имя выбранного профиля, его замеренная задержка и причина выбора.
        self.profile_name = ""
        self.profile_latency_ms: float | None = None
        self.profile_reason = ""
        self.profile_attempts: list[dict[str, Any]] = []

        # --- wiring to the event loop ---
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_json: SendJson | None = None
        self._send_bytes: SendBytes | None = None
        self._send_lock: asyncio.Lock | None = None

        # --- threads ---
        self._stop_event = threading.Event()
        self._capture_ready = threading.Event()
        self._new_frame = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None
        self._quick_thread: threading.Thread | None = None

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
        #: ТЗ F-311: (группа, зона) объектов внимания, о которых комната уже
        #: сказала хабу. Событие рождается на ПЕРЕХОДЕ, иначе «посылка у
        #: двери» приходило бы каждые полсекунды, пока посылка стоит.
        self._attention_seen: set[tuple[str, str]] = set()
        #: Находки внимания последнего кадра (ТЗ F-311): их читает
        #: ``_publish_attention`` сразу после ``_publish_state``.
        self._attention_found: list[dict[str, Any]] = []
        self._sent_state: tuple[int, tuple[tuple[str, int], ...]] | None = None
        self._sent_state_at = 0.0
        self._last_presence_push = 0.0
        #: ТЗ F-201: поднят, когда трек только что появился — такой проход
        #: нельзя ждать до следующего периодического кадра, он уходит сразу
        #: и целиком бёрстом (см. ``_maybe_push_presence``).
        self._burst_due = False
        #: ТЗ F-201: боксы, которые сторож увидел последним. Они уезжают в тот
        #: самый бёрст: без них человек спиной к камере не считается «в кадре»
        #: ни для правил хаба, ни для привязки лица к треку.
        self._quick_tracks: list[dict[str, Any]] = []
        #: Когда сторож последний раз забирал бёрст (monotonic, см. кулдаун).
        self._quick_burst_at = 0.0
        self._quick_errors = 0
        self._quick_count = 0
        #: Когда человека видели в последний раз (monotonic): пока он в
        #: комнате, детектор не ждёт лимит кадров (``_detection_budget``).
        self._last_person_seen = 0.0
        #: ТЗ F-306: необязательный слушатель кадров (жесты руки). Он получает
        #: кадр ПОСЛЕ отправки и не может ни задержать, ни сломать камеру.
        self._on_frame: Any = None
        #: ``event_id`` of the request being answered (ТЗ 4.5); empty for an
        #: unsolicited presence push, which mints its own id.
        self._request_event_id = ''

    def set_frame_listener(self, listener: Any) -> None:
        """Подписаться на каждый обработанный кадр (ТЗ F-306: жесты руки).

        Вызывается из потока YOLO, поэтому обработчик обязан быть быстрым и
        не бросать исключений: любая его ошибка глушится (см. `_infer_loop`).
        """
        self._on_frame = listener

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
    def privacy(self) -> PrivacyMode:
        """ТЗ F-303 mode. Built on demand, so a bare instance still answers.

        ``__init__`` builds it from ``client.camera.language``; an instance made
        without ``__init__`` (tests, one-off tooling) gets one here instead of an
        ``AttributeError`` in the middle of the frame path.
        """
        mode = getattr(self, "_privacy_mode", None)
        if mode is None:
            mode = self._privacy_mode = PrivacyMode("en")
        return mode

    @privacy.setter
    def privacy(self, mode: PrivacyMode) -> None:
        self._privacy_mode = mode

    # ------------------------------------------------------------------
    # frame zones (ТЗ F-309)
    # ------------------------------------------------------------------
    @property
    def zones(self) -> list[FrameZone]:
        """Зоны кадра, которые комната сейчас применяет (ТЗ F-309)."""
        return list(getattr(self, "_zones", ()))

    @property
    def masked_rev(self) -> str:
        """Отпечаток масок этого клиента; ``""`` — масок нет.

        Именно его хаб сверяет со своим конфигом: кадр считается
        замаскированным, только если комната закрасила те же области.
        """
        return masks_rev(getattr(self, "_zones", ()))

    def set_zones(self, raw: Any) -> bool:
        """Зоны кадра от хаба (``config_update``, ТЗ F-309).

        :returns: ``True``, если набор зон реально изменился (тогда следующий
            кадр уже понесёт новый отпечаток масок и хаб не откажет).
        """
        fresh = parse_zones(raw)
        current = list(getattr(self, "_zones", ()))
        if zones_rev(fresh) == zones_rev(current):
            return False
        self._zones = fresh
        masks = [zone for zone in fresh if zone.mask]
        log.info("Camera frame zones updated: %d zone(s), %d mask(s), rev %s",
                 len(fresh), len(masks), self.masked_rev or "-")
        return True

    @property
    def running(self) -> bool:
        """True while the capture thread is alive and delivering frames."""
        thread = self._capture_thread
        return bool(self._enabled and thread is not None and thread.is_alive())

    def set_privacy(self, on: bool, *, reason: str = "") -> bool:
        """ТЗ F-303: камера перестаёт смотреть (или снова смотрит).

        Включение сразу объявляет хабу, что камера выключена человеком: иначе
        хаб ждал бы кадров и путал бы «камеру выключили» с «в комнате никого»
        (F-301). Выключение сбрасывает дебаунс, чтобы первый же кадр ушёл
        сразу. Микрофон не затрагивается — иначе камеру нельзя было бы вернуть
        тем же голосом.
        """
        changed = self.privacy.set(on, reason=reason)
        if not changed:
            return False
        log.info("Camera privacy mode %s (%s)", "on" if self.privacy.on else "off",
                 reason or "voice")
        self._sent_state = None
        self._last_tracks_at = 0.0
        payload = {"type": MSG_CAMERA_STATE, "persons": 0, "tracks": [], "objects": {},
                   "privacy": bool(self.privacy.on)}
        if self._loop is not None:
            self._submit(self._send_state(payload))
        return True

    def announce_privacy(self) -> None:
        """Tell the hub the CURRENT privacy state, even when it did not change.

        A reconnected client says it again: the hub must not keep believing the
        camera is on just because the state was set before the link dropped.
        """
        if self._loop is None:
            return
        self._submit(self._send_state({
            "type": MSG_CAMERA_STATE, "persons": 0, "tracks": [], "objects": {},
            "privacy": bool(self.privacy.on)}))

    def _fail(self, message: str) -> None:
        """Disable the camera permanently, logging exactly one warning."""
        self._enabled = False
        self._stop_event.set()
        if not self._warned:
            self._warned = True
            log.warning("Camera disabled: %s. The voice assistant keeps working.", message)
        else:  # pragma: no cover - a second failure after the first warning
            log.debug("Camera failure after it was already disabled: %s", message)

    def _reconnect_capture(self, cv2: Any) -> Any:
        """Keep trying to open the device again until it answers or we stop.

        Deliberately endless: on a shared PC another program may hold the only
        webcam for hours, and the room has to come back on its own the moment it
        is free. The pause grows from :data:`CAMERA_RETRY_MIN_S` to
        :data:`CAMERA_RETRY_MAX_S`, so a dead device is not hammered and a
        camera that returns quickly is picked up in about two seconds.
        """
        delay = CAMERA_RETRY_MIN_S
        attempts = 0
        while not self._stop_event.is_set():
            if self._stop_event.wait(delay):
                return None
            try:
                capture = self._open_capture(cv2)
            except CameraUnavailable as exc:
                attempts += 1
                log.debug("Camera %d still unavailable (attempt %d): %s", self.index, attempts, exc)
                delay = min(delay * 2, CAMERA_RETRY_MAX_S)
                continue
            log.info("Camera %d reopened after %d failed attempt(s)", self.index, attempts)
            return capture
        return None

    def _report_health(self, ok: bool, detail: str) -> None:
        """Tell the hub the camera went away or came back (ТЗ F-702 extension).

        Владелец 2026-09-23: «в тг увед слать в группу уведов если чет не
        работает». The camera thread cannot await anything, so the frame is
        scheduled on the client's loop exactly like a presence state; the hub
        decides who hears about it. Never raises: a room without a hub must keep
        retrying silently.
        """
        send_json = self._send_json
        if send_json is None:
            return
        payload = {
            "type": MSG_ROOM_HEALTH,
            "kind": "camera",
            "ok": bool(ok),
            "detail": str(detail)[:200],
            "camera": f"usb:{self.index}" if not self.stream_url else "network",
        }
        try:
            self._submit(send_json(payload))
        except Exception as exc:  # noqa: BLE001 - a notice is never worth a crash
            log.debug("Could not report camera health: %s", exc)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        send_json: SendJson,
        send_bytes: SendBytes,
        send_lock: asyncio.Lock | None = None,
        on_unsent: Any = None,
    ) -> bool:
        """Start the capture and detection threads.

        :param loop: the client's event loop; the worker threads schedule their
            sends onto it with ``run_coroutine_threadsafe``.
        :param send_json: coroutine function sending one JSON control frame.
        :param send_bytes: coroutine function sending one binary frame.
        :param send_lock: optional lock held around every header+binary pair so
            camera frames cannot interleave with streamed microphone audio.
        :param on_unsent: ТЗ 4.8 — called with a presence frame the socket could
            not carry, so the client can deliver it after the reconnect.
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
        self._on_unsent = on_unsent

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
        if self._quick_model():
            self._quick_thread = threading.Thread(
                target=self._quick_gate_loop, name="jarvis-camera-quick", daemon=True
            )
            self._quick_thread.start()
        log.info(
            "Camera service starting: device index %d, %s at %.1f fps, "
            "one presence frame every %.1f s%s",
            self.index,
            self.model_name,
            self.fps,
            self.face_check_interval_s,
            (f", quick pass guard {self._quick_model()} at %.1f fps" % self.quick_fps)
            if self._quick_model() else " (quick pass guard off)",
        )
        return True

    def stop(self) -> None:
        """Stop both threads and release the device (safe to call twice)."""
        self._stop_event.set()
        self._capture_ready.set()
        self._new_frame.set()
        for thread in (self._quick_thread, self._infer_thread, self._capture_thread):
            if thread is None or not thread.is_alive():
                continue
            thread.join(timeout=JOIN_TIMEOUT_S)
            if thread.is_alive():  # pragma: no cover - a stuck driver call
                log.debug("Camera thread %s did not stop in time", thread.name)
        if (self._capture_thread is not None or self._infer_thread is not None
                or self._quick_thread is not None):
            log.info("Camera service stopped")
        self._capture_thread = None
        self._infer_thread = None
        self._quick_thread = None
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
                        # Владелец 2026-09-23: «там несколько камер… оно должно
                        # постоянно ретраить и в тг увед слать». Локальная камера
                        # раньше выключалась до конца процесса — на общем ПК
                        # устройство может уйти другому приложению или отвалиться
                        # на драйвере, и комната оставалась слепой до перезапуска.
                        # Теперь поток освобождает устройство, говорит хабу о
                        # поломке и открывает камеру заново с растущей паузой.
                        capture.release()
                        with self._frame_lock:
                            self._frame = None
                            self._frame_ts = 0.0
                            self._detected_frame = None
                        self._report_health(False, "the camera stopped delivering frames")
                        log.warning('Camera %d stopped delivering frames; reopening', self.index)
                        capture = self._reconnect_capture(cv2)
                        if capture is None:
                            return
                        failures = 0
                        self._report_health(True, 'the camera is delivering frames again')
                        log.info('Camera %d is working again', self.index)
                        continue
                    self._stop_event.wait(0.2)
                    continue
                failures = 0
                with self._frame_lock:
                    self._frame = frame
                    self._frame_ts = time.monotonic()
                    self._captured_count += 1
                # Владелец 2026-09-24: keep the recent past, so an alert clip
                # that starts a second after somebody walked past still shows
                # them (PREROLL_SECONDS). Cheap: a few small JPEGs per second.
                self._sample_preroll(frame)
                # read() waits for the device; extra sleeps reduce capture FPS.
                self._new_frame.set()
        finally:
            self._capture_ready.set()
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:  # pragma: no cover - teardown
                    log.debug("Error while releasing the camera: %s", exc)

    def _latest_frame(self) -> tuple[Any, float]:
        """The newest captured frame and its age in seconds (``None``, ``inf``)."""
        with self._frame_lock:
            frame = self._frame
            ts = self._frame_ts
        if frame is None:
            return None, float("inf")
        return frame, max(0.0, time.monotonic() - ts)

    def _latest_frame_ts(self) -> tuple[Any, float]:
        """The newest captured frame and its OWN monotonic timestamp (not age).

        Used by :meth:`_await_fresh_frame` to tell burst frames apart: an age
        alone cannot say whether two reads a few milliseconds apart landed on
        the same capture-thread frame or two different ones.
        """
        with self._frame_lock:
            return self._frame, self._frame_ts

    def _sample_preroll(self, frame: Any) -> None:
        """Keep one small JPEG of the last :data:`PREROLL_SECONDS` seconds.

        Runs in the capture thread, which must stay quick: one frame every
        ``1 / PREROLL_FPS`` is scaled to at most :data:`PREROLL_MAX_SIDE` and
        encoded once (a few milliseconds). The tracks of that moment are stored
        beside the picture, not drawn into it - the names come from the hub with
        the clip request, which happens later.
        """
        cv2 = self._cv2
        if cv2 is None or frame is None or not getattr(frame, "size", 0):
            return
        if not self.privacy.allows_frames:
            # Приватный режим: кадры комнаты не храним даже в памяти.
            return
        now = time.monotonic()
        if now - self._preroll_at < 1.0 / PREROLL_FPS:
            return
        self._preroll_at = now
        try:
            height, width = frame.shape[:2]
            scale = min(1.0, PREROLL_MAX_SIDE / float(max(width, height)))
            small = frame
            if scale < 1.0:
                small = cv2.resize(frame, (max(2, int(width * scale) // 2 * 2),
                                           max(2, int(height * scale) // 2 * 2)),
                                   interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return
        except Exception:  # noqa: BLE001 - the buffer must never hurt the camera
            log.debug("Could not keep a pre-roll frame", exc_info=True)
            return
        tracks = [dict(track) for track in list(getattr(self, "_tracks", []))]
        with self._preroll_lock:
            self._preroll.append((now, bytes(encoded), tracks))

    def preroll_frames(self, seconds: float) -> list[tuple[float, Any, list[dict[str, Any]]]]:
        """The stored frames of the last ``seconds``, oldest first.

        Each item is ``(monotonic, image, tracks)``; an unreadable JPEG is
        dropped instead of failing the clip. Called from the clip worker thread.
        """
        cv2 = self._cv2
        if cv2 is None or seconds <= 0:
            return []
        try:
            import numpy as np
        except Exception:  # pragma: no cover - numpy ships with the client
            return []
        cutoff = time.monotonic() - float(seconds)
        with self._preroll_lock:
            kept = [item for item in self._preroll if item[0] >= cutoff]
        frames: list[tuple[float, Any, list[dict[str, Any]]]] = []
        for at, encoded, tracks in kept:
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None and getattr(image, "size", 0):
                frames.append((at, image, [dict(track) for track in tracks]))
        return frames

    def _await_fresh_frame(self, newer_than: float) -> tuple[Any, float]:
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
    def _probe_frame(self) -> Any:
        """One frame for the F-312 measurement: the newest, or a blank one."""
        try:
            frame, _ = self._latest_frame_ts()
        except Exception:  # noqa: BLE001 - a missing capture must not block the choice
            frame = None
        if frame is not None:
            return frame
        try:
            import numpy as np

            return np.zeros((self.height, self.width, 3), dtype="uint8")
        except Exception:  # noqa: BLE001 - no numpy, no synthetic frame
            return None

    def _select_profile(self, yolo_class: Any) -> Any:
        """ТЗ F-312: pick the profile by measured latency; return its model.

        Each candidate is loaded and timed on the same frame; the first profile
        whose median inference fits its budget wins, and a profile whose
        measurement fails is skipped with the reason. ``None`` means nothing
        could be measured or chosen — the caller then loads the configured model
        exactly as before, so a broken measurement never costs the client its
        camera.
        """
        from client.vision_profile import choose_profile, measure_ms, plan_profiles

        candidates, skipped = plan_profiles(self._camera_cfg)
        self.profile_attempts = [
            {"profile": item.name, "latency_ms": item.latency_ms, "reason": item.reason}
            for item in skipped
        ]
        for item in skipped:
            log.info("Detection profile %s skipped (%s)", item.name, item.reason)
        if not candidates:
            self.profile_reason = "нет доступных профилей"
            log.warning("No detection profile can be measured; using %s", self.model_name)
            return None
        frame = self._probe_frame()
        if frame is None:
            self.profile_reason = "нет кадра для замера"
            log.warning("No frame to measure a detection profile on; using %s", self.model_name)
            return None
        loaded: dict[str, Any] = {}

        def measure(profile: Any) -> float:
            try:
                model = yolo_class(profile.model)
                loaded[profile.name] = model
                return measure_ms(lambda: model.predict(source=frame, imgsz=640,
                                                        device=self.device,
                                                        conf=CONF_THRESHOLD,
                                                        verbose=False,
                                                        **precision_kwargs(
                                                            bool(profile.half) and self.half)),
                                  frames=self.profile_measure_frames)
            except Exception as exc:  # noqa: BLE001 - записанная причина важнее трейсбека
                self.profile_attempts.append(
                    {"profile": profile.name, "latency_ms": None,
                     "reason": f"загрузка/замер не удались: {type(exc).__name__}: {exc}"})
                raise

        choice = choose_profile(candidates, measure)
        if choice is None:
            self.profile_reason = "замер не удался ни на одном профиле"
            log.warning("No detection profile could be measured; using %s", self.model_name)
            return None
        self.model_name = choice.profile.model
        self.half = choice.profile.half
        self.fps = (0. if choice.profile.fps == 0
                    else min(MAX_FPS, max(MIN_FPS, choice.profile.fps)))
        self.profile_name = choice.profile.name
        self.profile_latency_ms = choice.latency_ms
        self.profile_reason = choice.reason
        self.profile_attempts.extend(
            {"profile": item.name, "latency_ms": item.latency_ms, "reason": item.reason}
            for item in choice.attempts
        )
        log.info("Detection profile chosen by measurement: %s", choice.text())
        return loaded.get(choice.profile.name)

    def _infer_loop(self) -> None:
        self._capture_ready.wait()
        if self._stop_event.is_set() or not self._enabled:
            return
        try:
            yolo_class = self._import_yolo()
        except CameraUnavailable as exc:
            self._fail(str(exc))
            return
        # ТЗ F-312: с включённым автовыбором модель подбирается ЗАМЕРОМ, а не
        # по имени файла; выключенный флаг оставляет ровно прежнее поведение.
        model = self._select_profile(yolo_class) if self.auto_profile else None
        if model is None:
            try:
                model = yolo_class(self.model_name)
            except Exception as exc:  # noqa: BLE001 - weights download / CUDA errors
                self._fail(f"the YOLO model {self.model_name!r} could not be loaded: {exc}")
                return
        log.info('YOLO model %s loaded, device=%s, FP16=%s, FPS limit=%s%s',
                 self.model_name, self.device, self.half, self.fps or 'unlimited',
                 (f', profile {self.profile_name}'
                  f' ({self.profile_latency_ms:.0f} ms/frame, {self.profile_reason})')
                 if self.profile_name else '')
        # ТЗ F-201: BoT-SORT's memory is counted in FRAMES, so the thirty
        # seconds of re-association the ТЗ asks for are computed from the real
        # frame rate instead of being left at the shipped default.
        try:
            self.tracker_path = write_tracker_config(
                Path(__file__).with_name('room-tracker.runtime.yaml'),
                self.fps or 10.0)
        except Exception as exc:  # noqa: BLE001 - the shipped config still works
            self.tracker_path = Path(__file__).with_name('room-tracker.yaml')
            log.warning('Could not write the runtime tracker config (%s); using %s',
                        exc, self.tracker_path.name)
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
            if not self.privacy.allows_frames:
                # ТЗ F-303: пока камера «не смотрит», детектор не запускается
                # вообще — ни людей, ни объектов, ни кропов.
                self._track_reports = []
                self._tracks = []
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    self._stop_event.wait(remaining)
                continue
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
                    if persons >= 1:
                        self._last_person_seen = time.monotonic()
                    self._cache_detection(frame, frame_ts)
                    self._publish_state(persons, objects)
                    self._publish_attention(getattr(self, '_attention_found', []))
                    self._publish_tracks()
                    self._publish_body_crops(frame)
                    listener = getattr(self, '_on_frame', None)
                    if listener is not None:
                        # ТЗ F-306: жесты руки считаются на этом же кадре, но их
                        # ошибка не имеет права уронить камеру или ход.
                        try:
                            listener(frame)
                        except Exception as exc:  # noqa: BLE001
                            log.debug("Frame listener failed (%s)", exc)
                    if persons >= 1:
                        if self._frame_recorder is not None:
                            self._frame_recorder.submit(frame, time.time() - (time.monotonic() - frame_ts),
                                {'persons': persons, 'tracks': list(self._tracks), 'objects': objects,
                                 'camera_id': 'network' if self.stream_url else f'usb:{self.index}',
                                 'model': self.model_name}, stop_event=self._stop_event)
                        self._maybe_push_presence(frame)
                    self._report_performance()
            remaining = self._detection_budget(interval) - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _detection_budget(self, interval: float) -> float:
        """Seconds to wait before the next detection: none while people are here.

        ``interval`` comes from the configured/measured FPS. Spending it while a
        person is visible drops the very frames that identify them, so the
        detector runs flat out from a sighting until ``ACTIVE_DETECTION_HOLD_S``
        after the last one.
        """
        active = (bool(getattr(self, '_tracks', None))
                  or time.monotonic() - float(getattr(self, '_last_person_seen', 0.0))
                  <= ACTIVE_DETECTION_HOLD_S)
        return 0.0 if active else interval

    # ------------------------------------------------------------------
    # quick pass guard (ТЗ F-201)
    # ------------------------------------------------------------------
    def _quick_model(self) -> str:
        """The light guard's weights, or ``""`` when the guard is off.

        Off means: switched off in the config, no rate, or the room's own
        detector is already that light model - then there is nothing to add.
        """
        name = str(getattr(self, 'quick_model_name', '') or '').strip()
        if not name or not getattr(self, 'quick_fps', 0.0):
            return ''
        return '' if name == self.model_name else name

    def _quick_gate_loop(self) -> None:
        """Watch for the first sign of a person between the heavy frames (F-201).

        ON THE OWNER'S PC (2026-09-22): ``yolo11x`` needs ~300 ms per frame, so
        the accurate detector sees the room roughly three times a second - a
        person who walks past in half a second may never appear in one of its
        frames, and no later stage can report somebody nobody saw. This thread
        runs the LIGHT model on the frames the capture thread already has and,
        on the first sighting while no track exists, releases the very next
        presence burst with those boxes: the hub gets the person's picture (and
        their face) while they are still in the room.

        The guard never decides who it was, never publishes state, and never
        throws into the camera: if the light weights cannot be loaded or keep
        failing, the room simply works the way it did before this thread.
        """
        self._capture_ready.wait()
        if self._stop_event.is_set() or not self._enabled:
            return
        name = self._quick_model()
        if not name:
            return
        try:
            yolo_class = self._import_yolo()
            model = yolo_class(name)
        except Exception as exc:  # noqa: BLE001 - the guard is optional
            log.warning("Quick pass guard is off: %s could not be loaded (%s)", name, exc)
            return
        log.info("Quick pass guard running: %s at up to %.1f fps between the %s frames",
                 name, self.quick_fps, self.model_name)
        interval = 1.0 / self.quick_fps if self.quick_fps else 0.0
        last_frame_ts = 0.0
        while not self._stop_event.is_set():
            if self._quick_model() != name:
                # Автовыбор профиля (ТЗ F-312) перевёл тяжёлый детектор на те
                # же лёгкие веса — второй раз смотреть ими незачем.
                log.info("Quick pass guard stops: the detector now runs %s itself", name)
                return
            started = time.monotonic()
            frame, frame_ts = self._latest_frame_ts()
            if frame is None or frame_ts <= last_frame_ts:
                self._stop_event.wait(0.02)
                continue
            last_frame_ts = frame_ts
            if self.privacy.allows_frames and started - frame_ts <= STALE_FRAME_S:
                try:
                    boxes = self._quick_person_boxes(model, frame, frame_ts)
                except Exception as exc:  # noqa: BLE001 - a broken guard is not a broken camera
                    self._quick_errors += 1
                    if self._quick_errors == 1:
                        log.warning("The quick pass guard failed (%s)", exc)
                    else:
                        log.debug("The quick pass guard failed (%d): %s", self._quick_errors, exc)
                    if self._quick_errors >= 10:
                        log.warning("Quick pass guard stopped after %d failures", self._quick_errors)
                        return
                    self._stop_event.wait(0.5)
                    continue
                self._quick_errors = 0
                self._quick_count += 1
                if boxes:
                    self._note_quick_persons(frame, boxes)
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _quick_person_boxes(self, model: Any, frame: Any, frame_ts: float) -> list[dict[str, Any]]:
        """Person boxes the light model sees in one frame, as wire tracks."""
        results = model.predict(
            source=frame, conf=QUICK_PASS_CONF, imgsz=QUICK_PASS_IMGSZ,
            device=self.device, verbose=False, **precision_kwargs(self.half),
        )
        return self._person_boxes(results, prefix=f'quick:{int(frame_ts * 1000)}')

    @staticmethod
    def _person_boxes(results: Any, *, prefix: str = 'quick') -> list[dict[str, Any]]:
        """Read ``person`` boxes out of a plain ``predict`` result (never raises).

        The hub reads either wire shape of a track, so the guard's boxes go out
        as the v1.4 ``{"id", "box"}`` rows a ``camera_state`` already carries.
        """
        found: list[dict[str, Any]] = []
        for result in results or []:
            names = getattr(result, "names", None) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            classes = getattr(boxes, "cls", None)
            confidences = getattr(boxes, "conf", None)
            positions = getattr(boxes, "xyxyn", None)
            if classes is None or confidences is None or positions is None:
                continue
            class_list = classes.tolist() if hasattr(classes, "tolist") else list(classes)
            conf_list = confidences.tolist() if hasattr(confidences, "tolist") else list(confidences)
            position_list = positions.tolist() if hasattr(positions, "tolist") else list(positions)
            for index, (class_id, confidence) in enumerate(zip(class_list, conf_list)):
                if float(confidence) < QUICK_PASS_CONF or index >= len(position_list):
                    continue
                class_index = int(class_id)
                label = str(names.get(class_index, class_index) if isinstance(names, dict) else class_index)
                if label != 'person':
                    continue
                x1, y1, x2, y2 = (max(0.0, min(1.0, float(value)))
                                  for value in position_list[index][:4])
                if x2 <= x1 or y2 <= y1:
                    continue
                found.append({"id": f"{prefix}:{len(found)}", "box": [x1, y1, x2, y2]})
        return found

    def _quick_burst_due(self, now: float) -> bool:
        """Should the guard take the next presence burst? (no track, not just now)."""
        if getattr(self, '_tracks', None):
            # The heavy detector already owns these frames (ТЗ F-201).
            return False
        if self._presence_pending.is_set():
            return False
        cooldown = float(getattr(self, '_quick_burst_at', 0.0)) + QUICK_PASS_COOLDOWN_S
        return now >= cooldown

    def _note_quick_persons(self, frame: Any, boxes: list[dict[str, Any]]) -> None:
        """The guard saw somebody: hand the next presence burst to them (F-201)."""
        now = time.monotonic()
        self._last_person_seen = now
        if not self._quick_burst_due(now):
            return
        self._quick_burst_at = now
        self._burst_due = True
        self._quick_tracks = list(boxes)
        log.info('Quick pass guard saw %d person(s) - taking the presence burst now',
                 len(boxes))
        try:
            self._maybe_push_presence(frame, frame_prefix='q')
        except Exception as exc:  # noqa: BLE001 - the guard never breaks the camera
            log.debug('Could not push the guard burst (%s)', exc)

    def _detect(self, model: Any, frame: Any) -> tuple[int, dict[str, int], list[dict[str, Any]]]:
        """Count people and objects in one frame by YOLO class name.

        Возвращает ещё и объекты внимания (ТЗ F-311) с зоной кадра: ``label``
        — каноническая группа (кошка/собака/посылка), ``zone`` — имя зоны
        дома, в которую попал центр находки. Объект, попавший в область «не
        анализировать» (F-309), сюда не попадает: маску закрашивают, чтобы её
        не смотрели, и сообщать о находке внутри неё было бы тем же
        смотрением, только словами.
        """
        tracker = getattr(self, 'tracker_path', None) or str(
            Path(__file__).with_name('room-tracker.yaml'))
        results = model.track(
            source=frame,
            persist=True,
            tracker=str(tracker),
            conf=CONF_THRESHOLD,
            verbose=False,
            device=self.device,
            imgsz=640,
            **precision_kwargs(self.half),
        )
        counts: dict[str, int] = {}
        detections = []
        attention: list[dict[str, Any]] = []
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
                    detections.append({
                        'track_id': f'{self._track_epoch}:{int(ids[box_index])}',
                        'bbox': [max(0., min(1., float(v))) for v in positions[box_index]],
                        'conf': float(confidence),
                    })
                    continue
                group = attention_group(label)
                if not group or box_index >= len(positions):
                    continue
                x1, y1, x2, y2 = (max(0., min(1., float(v))) for v in positions[box_index][:4])
                centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                zones = getattr(self, '_zones', ())
                if masked_at(zones, *centre):
                    continue
                attention.append({'label': group, 'zone': zone_at(zones, *centre),
                                  'conf': float(confidence)})
        # ТЗ F-201: the registry keeps the id of a person who stepped out of
        # the frame for thirty seconds, so coming back is a re-association.
        self._track_reports = self.tracks.observe(detections, now=time.monotonic())
        self._tracks = [{'id': report.track_id,
                         'box': [float(value) for value in report.bbox]}
                        for report in self._track_reports]
        persons = counts.pop("person", 0)
        # ТЗ F-311: находки внимания читает `_publish_attention`; отдельным
        # полем, а не третьим значением, чтобы `_detect` остался тем же
        # вызовом, что и раньше (клиенты и тесты зовут его как пару).
        self._attention_found = attention
        # Владелец 2026-09-24: the live preview shows EVERY box of the last
        # inference, not only people and attention groups. Kept as plain floats
        # so the drawing thread never touches torch tensors.
        drawn: list[dict[str, Any]] = []
        for box_index, (class_id, confidence) in enumerate(zip(class_list, conf_list)):
            if float(confidence) < CONF_THRESHOLD or box_index >= len(positions):
                continue
            index = int(class_id)
            label = str(names.get(index, index) if isinstance(names, dict) else index)
            coords = [max(0., min(1., float(v))) for v in positions[box_index][:4]]
            if len(coords) != 4:
                continue
            drawn.append({'label': label, 'box': coords, 'conf': float(confidence)})
            if len(drawn) >= LIVE_PREVIEW_MAX_DETECTIONS:
                break
        self._last_detections = drawn
        return int(persons), counts

    def _report_performance(self):
        now = time.monotonic()
        elapsed = now - self._metrics_at
        if elapsed < 5:
            return
        stats = {'ts': time.time(), 'model': self.model_name, 'half': self.half,
                 'fps_limit': self.fps,
                 # ТЗ F-312: чем закончился автовыбор профиля (пусто, если он
                 # выключен) — видно и в логе, и в data/camera-performance.json.
                 'profile': self.profile_name or None,
                 'profile_latency_ms': self.profile_latency_ms,
                 'profile_reason': self.profile_reason or None,
                 'profile_attempts': list(self.profile_attempts),
                 'capture_fps': round((self._captured_count - self._metrics_captured) / elapsed, 2),
                 'yolo_fps': round((self._inferred_count - self._metrics_inferred) / elapsed, 2),
                 'captured_frames': self._captured_count, 'processed_frames': self._inferred_count,
                 # ТЗ F-201: how the light guard is doing - the number that says
                 # whether a quick pass-by had anything to be caught by at all.
                 'quick_guard': self._quick_model() or None,
                 'quick_frames': self._quick_count,
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
    def _publish_state(self, persons: int, objects: dict[str, int]) -> None:
        """Send ``camera_state`` when the picture changed, at most every 2 s.

        YOLO flickers object counts (a bottle drifting between 1 and 2 across
        frames), which used to re-announce the "change" every debounce window.
        A PERSON-count change is reported at once; an object-only change must
        stay identical for two consecutive checks before it is sent.
        """
        if not self.privacy.allows_frames:
            # ТЗ F-303: камера не смотрит — хабу нечего сообщать о комнате.
            return
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

    async def _send_state(self, payload: dict[str, Any]) -> None:
        send_json = self._send_json
        if send_json is None:
            return
        try:
            await send_json(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead socket is normal here
            self._note_send_failure("camera_state", exc)
            self._keep_for_replay(payload)

    def _keep_for_replay(self, payload: Any) -> None:
        """ТЗ 4.8: the frame never left the room — the client may replay it.

        Only presence states are offered (the client's own buffer decides what
        it keeps): the room is not going to hand five-minute-old pictures to
        the hub, which would describe a past room as if it were the present.
        """
        callback = getattr(self, "_on_unsent", None)
        if callback is None or not isinstance(payload, dict):
            return
        try:
            callback(dict(payload))
        except Exception as exc:  # noqa: BLE001 - a full buffer is not a crash
            log.debug("Could not buffer the unsent %s (%s)", payload.get("type"), exc)

    def _publish_attention(self, attention: list[dict[str, Any]]) -> None:
        """ТЗ F-311: сказать хабу о ПОЯВИВШЕМСЯ объекте внимания и его зоне.

        Событие рождается на переходе «пары (объект, зона) не было — пара
        появилась», поэтому правило дома не срабатывает каждые полсекунды.
        Комната не выдумывает зону: её посчитал `_detect` по полигонам дома
        (F-309), а объект, попавший в маску, сюда вообще не доходит.
        """
        if not self.privacy.allows_frames:
            # ТЗ F-303: приватный режим — камера не смотрит, и событий нет.
            return
        try:
            current = {(str(item.get('label') or ''), str(item.get('zone') or ''))
                       for item in attention or [] if str(item.get('label') or '')}
        except (TypeError, AttributeError):
            return
        previous = getattr(self, '_attention_seen', set())
        self._attention_seen = current
        for item in attention or []:
            label = str(item.get('label') or '')
            key = (label, str(item.get('zone') or ''))
            if not label or key in previous:
                continue
            self._submit(self._send_object_event(label, key[1], item.get('conf')))

    async def _send_object_event(self, label: str, zone: str, confidence: Any) -> None:
        """One ``object_event`` frame (ТЗ F-311): что, где и насколько уверенно."""
        send_json = self._send_json
        if send_json is None:
            return
        payload = {
            "type": MSG_OBJECT_EVENT,
            "label": str(label)[:40],
            "zone": str(zone or "")[:120],
            "conf": round(_as_float(confidence, 0.0), 3),
            "at_ms": int(time.time() * 1000),
            "event_id": new_ulid(),
        }
        try:
            await send_json(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead socket is normal here
            self._note_send_failure("object_event", exc)

    def _publish_tracks(self) -> None:
        """ТЗ F-201: the room's own person tracks, in their own message.

        The ids are what the hub reasons about ("the same person came back"),
        so a changed SET of tracks is announced at once and the boxes are
        refreshed with the same debounce the person count already uses. A
        client that predates this message keeps sending ``camera_state``.
        """
        if not self._tracks_message:
            return
        if not self.privacy.allows_frames:
            return
        reports = list(getattr(self, '_track_reports', []))
        if any(report.event in {'entered', 'returned'} for report in reports):
            # Somebody just walked in. A quick pass-by is over before the
            # periodic presence push would carry a second frame, so the very
            # next push goes out at once and carries the whole burst (see
            # ``_maybe_push_presence``).
            self._burst_due = True
        ids = tuple(sorted(report.track_id for report in reports))
        now = time.monotonic()
        if ids == self._sent_track_ids and now - self._sent_tracks_at < STATE_DEBOUNCE_S:
            return
        self._sent_track_ids = ids
        self._sent_tracks_at = now
        for report in reports:
            if report.event in {'entered', 'returned'}:
                log.info('Track %s %s (gap %.1f s, %.0f%% conf)',
                         report.track_id, report.event, report.gap_s, 100 * report.conf)
        payload = {
            "type": MSG_TRACKS,
            "tracks": [report.as_wire() for report in reports],
        }
        self._submit(self._send_state(payload))

    def _publish_body_crops(self, frame: Any) -> None:
        """ТЗ F-202: cut a crop of every track that needs one right now.

        Full height, at most 640 px tall, cut HERE - the hub never sees the
        room's full frames unless it asks for one. Cut on appearance, every two
        seconds and whenever the person turned (the box's aspect changed).
        """
        if not self._tracks_message or getattr(self, '_cv2', None) is None:
            return
        if not self.privacy.allows_frames:
            return
        try:
            # ТЗ F-309: кроп — та же картинка комнаты, поэтому маска
            # закрашивается до вырезки: иначе область «не анализировать»
            # уехала бы в хаб внутри кропа.
            frame = self._mask_frame(frame, self._cv2)
        except CameraUnavailable as exc:
            log.warning("Not sending body crops from this frame: %s", exc)
            return
        now = time.monotonic()
        for report in list(getattr(self, '_track_reports', [])):
            x1, y1, x2, y2 = report.bbox
            aspect = max(1e-6, abs(x2 - x1)) / max(1e-6, abs(y2 - y1))
            if not self._crop_schedule.should_send(report.track_id, aspect, now=now):
                continue
            encoded = encode_crop(frame, report.bbox, self._cv2)
            if encoded is None:
                continue
            jpeg, width, height = encoded
            self._submit(self._send_body_crop(report.track_id, jpeg, width, height))

    async def _send_body_crop(self, track_id: str, jpeg: bytes, width: int,
                              height: int) -> None:
        send_json, send_bytes = self._send_json, self._send_bytes
        if send_json is None or send_bytes is None:
            return
        lock = self._send_lock
        if lock is not None and lock.locked():
            # The microphone is streaming: the server would read this JPEG as
            # audio. The next two-second window carries the same person.
            log.debug("Skipping a body crop - the socket is busy")
            return
        masked_rev = self.masked_rev
        header = {"type": MSG_BODY_CROP, "track_id": track_id, "kind": "body",
                  "w": int(width), "h": int(height),
                  # ТЗ F-309: маска закрашена до вырезки, хаб это проверяет.
                  "masked": bool(masked_rev), "zones_rev": masked_rev}
        try:
            if lock is None:
                await send_json(header)
                await send_bytes(jpeg)
            else:
                async with lock:
                    await send_json(header)
                    await send_bytes(jpeg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconnects are routine
            self._note_send_failure("body_crop", exc)

    # ------------------------------------------------------------------
    # camera_frame (SPEC v1.4)
    # ------------------------------------------------------------------
    def _maybe_push_presence(self, frame=None, frame_prefix: str = 'p') -> None:
        """Push a :data:`FACE_BURST`-frame burst while somebody is visible (v1.4 burst).

        JPEG encoding runs off the YOLO thread. At most one presence update is
        pending; its track boxes are captured together with its source frame.
        A track that just appeared is not held back by the periodic interval:
        somebody who crosses the room in half a second gets their frames taken
        right then, and that push carries the whole :data:`FACE_BURST`.

        :param frame_prefix: ``p`` for the detector's own pushes, ``q`` for the
            light guard's (ТЗ F-201). The hub groups one burst by its frame id,
            so two threads must never mint the same id.
        """
        now = time.monotonic()
        if not self.privacy.allows_frames:
            # ТЗ F-303: в приватном режиме не уходит даже «присутствие».
            return
        burst = bool(getattr(self, '_burst_due', False))
        if self._presence_pending.is_set() or (
                not burst and now - self._last_presence_push < self.face_check_interval_s):
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
        self._burst_due = False
        # ТЗ F-201: a burst the light guard took carries the guard's own boxes -
        # a person whose back is turned still counts as "somebody in the frame"
        # for the hub's rules. The heavy detector's tracks win when it has any.
        tracks = list(self._tracks) or list(getattr(self, '_quick_tracks', ()) or ())
        self._quick_tracks = []
        self._frame_seq += 1
        frame_id = f'{frame_prefix}{self._frame_seq}'
        # ТЗ 4.5: an unprompted presence push is a background camera event too,
        # so the client mints its own event id for the burst.
        event_id = new_ulid()
        self._presence_pending.set()
        if self._submit(self._encode_and_push_presence(frame, frame_id, tracks,
                                                       event_id, burst=burst)) is None:
            self._presence_pending.clear()

    async def _encode_and_push_presence(self, frame, frame_id, tracks, event_id='', burst=False):
        try:
            # full=True: face recognition needs a CRISP face. The old 1280px /
            # quality-80 presence frame made the owner's own face score around
            # the match threshold; a native-resolution frame fixes that at the
            # source instead of lowering the bar.
            if burst:
                # A person who just appeared may be gone in half a second: the
                # first push carries the whole burst the module promises, so
                # the hub gets the several frames its identity and appearance
                # confirmation asks for instead of one.
                pairs = await asyncio.to_thread(self._capture_burst_sync, FACE_BURST, full=True)
            else:
                pairs = ([await asyncio.to_thread(self._encode, frame, full=True)]
                         if frame is not None
                         else await asyncio.to_thread(self._capture_burst_sync, 1, full=True))
            if pairs:
                await self._send_burst(frame_id, CAMERA_REASON_PRESENCE, pairs, skip_if_busy=True,
                                       tracks=tracks, event_id=event_id)
        except Exception as exc:  # noqa: BLE001 - capture/encoding must not kill the thread
            log.debug("Could not capture the presence burst: %s", exc)
            return
        finally:
            self._presence_pending.clear()

    @staticmethod
    def _frame_size(frame: Any) -> tuple[int, int]:
        """``(width, height)`` кадра OpenCV; ``(0, 0)`` — если это не картинка."""
        try:
            return int(frame.shape[1]), int(frame.shape[0])
        except (AttributeError, IndexError, TypeError):
            return 0, 0

    def _mask_frame(self, frame: Any, cv2: Any) -> Any:
        """ТЗ F-309: закрасить области «не анализировать» ДО JPEG.

        Копия обязательна: кадр приходит из кэша камеры, и затирание «на
        месте» испортило бы следующую картинку (и кадр трекера). Если масок
        нет, кадр возвращается как есть — без лишней копии и без изменений.

        Закрашивание в чёрное, а не «вырезание»: JPEG не умеет дырок, а
        чёрный прямоугольник гарантирует, что пикселей области в кадре нет.
        Ошибка тут не «мелкая»: не закрасив маску, комната прислала бы то,
        что владелец просил не смотреть, поэтому кадр не отправляется вовсе.
        """
        width, height = self._frame_size(frame)
        polygons = mask_polygons(getattr(self, "_zones", ()), width, height)
        if not polygons:
            return frame
        try:
            import numpy as np

            masked = frame.copy()
            for polygon in polygons:
                cv2.fillPoly(masked, [np.array(polygon, dtype=np.int32)], (0, 0, 0))
            return masked
        except Exception as exc:  # noqa: BLE001 - молчать о маске нельзя
            raise CameraUnavailable(f"could not paint the frame mask: {exc}") from exc

    def _encode(self, frame: Any, full: bool = False) -> tuple[bytes, int, int]:
        """Downscale to :data:`MAX_SIDE_PX` and encode as JPEG q80.

        :param full: v1.6 -- skip the downscale entirely for this frame
            (``find_object`` wants the detector to see native resolution).

        ТЗ F-309: области «не анализировать» закрашиваются ЗДЕСЬ, до JPEG, —
        иначе маска осталась бы обещанием, а не свойством кадра.
        """
        cv2 = self._cv2
        if cv2 is None:  # pragma: no cover - only reachable before the first frame
            raise CameraUnavailable("OpenCV is not loaded")
        frame = self._mask_frame(frame, cv2)
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
        event_id: str = "",
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
        # ТЗ F-309: кадр уходит уже с закрашенными масками, и заголовок несёт
        # отпечаток этих масок — хаб сверяет его со своим конфигом и
        # отказывается анализировать кадр без маски.
        masked_rev = self.masked_rev
        log.debug(
            "Sending a %s frame burst %s: %d frame(s)", reason, frame_id, total
        )
        try:
            if lock is None:
                for seq, pair in enumerate(pairs, start=1):
                    jpeg, width, height = pair[:3]
                    frame_tracks = pair[3] if len(pair) > 3 else tracks
                    await self._send_pair(
                        send_json, send_bytes, frame_id, reason, seq, total, jpeg, width, height,
                        frame_tracks, event_id, masked=bool(masked_rev), zones_rev=masked_rev
                    )
            else:
                async with lock:
                    for seq, pair in enumerate(pairs, start=1):
                        jpeg, width, height = pair[:3]
                        frame_tracks = pair[3] if len(pair) > 3 else tracks
                        await self._send_pair(
                            send_json, send_bytes, frame_id, reason, seq, total, jpeg, width, height,
                            frame_tracks, event_id, masked=bool(masked_rev), zones_rev=masked_rev
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
        event_id: str = "",
        masked: bool = False,
        zones_rev: str = "",
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
            # ТЗ F-309: маска уже закрашена в самом JPEG; заголовок говорит
            # хабу, что именно проверять.
            "masked": bool(masked),
            "zones_rev": str(zones_rev or ""),
        }
        if event_id:
            # ТЗ 4.5: the background camera event keeps the id the server minted
            # (or the one this client minted for an unprompted presence push).
            header["event_id"] = event_id
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
            payload = {"type": MSG_CAMERA_ERROR, "id": str(request_id), "error": str(error)}
            if self._request_event_id:
                payload["event_id"] = self._request_event_id
            await send_json(payload)
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
    async def serve_request(self, request_id: str, burst: int = 1, full: bool = False,
                            event_id: str = "") -> None:
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
        # ТЗ 4.5: the server's event id is echoed on every header of the answer.
        self._request_event_id = str(event_id or '')[:100]
        if not self._enabled:
            await self._send_error(request_id, "the camera is not available on this client")
            return
        if not self.privacy.allows_frames:
            # ТЗ F-303: приватный режим — это не «нет кадров из-за ошибки», а
            # прямое «камера выключена человеком»; ничего не снимаем и не шлём.
            await self._send_error(request_id, "the camera is off (privacy mode)")
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
            "Answering the camera request (id=%s, event %s): %d/%d frame(s)",
            request_id or "?", self._request_event_id or "-", len(pairs), count,
        )
        await self._send_burst(request_id, CAMERA_REASON_REQUEST, pairs,
                               event_id=self._request_event_id)

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
