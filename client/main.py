"""Jarvis room client — entry point (SPEC §7).

Single asyncio loop:

1. connect to the brain server over WebSocket (auto-reconnect, 3 s backoff,
   ``hello`` re-sent every time),
2. listen for the wake word with Vosk, play a generated ack beep,
3. record the utterance with webrtcvad, including the ``pre_roll_ms`` of audio
   captured before the trigger,
4. stream it to the server in 30 ms chunks and handle every server message,
5. run the actions through :mod:`client.actions.dispatcher` and report results
   (``ok``/``error``/``output``), and answer ``screenshot_request`` with a JPEG
   grabbed by :mod:`client.screen`,
6. play the TTS stream as it arrives,
7. optionally keep listening for ``followup_window_s`` seconds without the
   wake word.

Per utterance the server may send several rounds of ``actions`` and/or
``screenshot_request`` (one per LLM tool round) before the spoken reply, so the
response loop keeps handling messages until ``tts_end`` or ``error``.

v1.4 — the socket is read ALL the time
--------------------------------------
The server talks between utterances too: it greets a face it does not know and
it pulls camera frames for ``look_at_camera``/``enroll_face``. So exactly one
background task (:meth:`JarvisClient._reader_loop`) owns ``ws.recv()`` for the
lifetime of a connection and routes what it reads:

* ``camera_request`` — answered at any moment from :mod:`client.camera`;
* ``image_show`` (v1.6) — its header + single binary JPEG are handled
  directly by the reader in EITHER mode and handed to :mod:`client.viewer`,
  never queued as a conversation message or mistaken for TTS/mic audio;
* during a conversation — everything else (binary frames included) goes into
  an ``asyncio.Queue`` that :meth:`JarvisClient._receive_response` consumes;
* while idle — a proactive ``say`` + ``tts_start``…``tts_end`` block is played
  through the speakers, and the wake word cuts it short and starts listening.

The mode flag is owned by the conversation loop and flips exactly at
``utterance_start``/end of reply, so no message is ever consumed twice or lost
in between. The reader dies with the connection and is restarted by
:meth:`JarvisClient._ensure_link` after every reconnect; the camera keeps
running across reconnects and resumes its pushes by itself.

Run from the repository root: ``python -m client.main --config config.yaml``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import protocol as _protocol
from common.client_config import load_client_config as load_config
from common.protocol import (
    MSG_ACTION_RESULT,
    MSG_ACTIONS,
    MSG_CAMERA_ERROR,
    MSG_CAMERA_REQUEST,
    MSG_ERROR,
    MSG_HELLO,
    MSG_IMAGE_SHOW,
    MSG_READY,
    MSG_SAY,
    MSG_SCREENSHOT,
    MSG_SCREENSHOT_ERROR,
    MSG_SCREENSHOT_REQUEST,
    MSG_SPEAKER,
    MSG_STATUS,
    MSG_TRANSCRIPT,
    MSG_TTS_END,
    MSG_TTS_START,
    MSG_UTTERANCE_END,
    MSG_UTTERANCE_START,
)

from client.actions.dispatcher import Dispatcher
from client.attention import followup_seconds
from client.audio import (
    BEEP_FREQ_HZ,
    BEEP_MS,
    ERROR_BEEP_FREQ_HZ,
    ERROR_BEEP_MS,
    FRAME_MS,
    SAMPLE_WIDTH,
    AudioInput,
    AudioOutput,
    RingBuffer,
    frame_bytes,
)
from client.devices.registry import build_registry
from client.screen import SCREENSHOT_FORMAT, Capture, capture_jpeg
from client.vad import VadRecorder
from client.wakeword import WakeWordDetector
from client.voice_controls import ConfirmedWakeDetector, SilenceDetector
from common.voice_commands import mentions_silence_command
from client.ws_client import WAIT_FOREVER, WSClient, WSDisconnected

log = logging.getLogger("client")

#: The camera stack (OpenCV + Ultralytics) is optional and lives behind lazy
#: imports, but even importing this thin module must not be able to stop the
#: voice client — a broken checkout simply means "no camera on this machine".
_CAMERA_IMPORT_ERROR: Optional[str] = None
try:
    from client.camera import CameraService
except Exception as _camera_exc:  # noqa: BLE001 - pragma: no cover
    _CAMERA_IMPORT_ERROR = f"{type(_camera_exc).__name__}: {_camera_exc}"
    CameraService = None  # type: ignore[assignment]

#: v1.6: the detections-photo viewer only lazily touches cv2 (inside its own
#: methods), but the import is still guarded the same way as the camera
#: stack — a broken checkout must never be able to stop the voice client.
_VIEWER_IMPORT_ERROR: Optional[str] = None
try:
    from client.viewer import ImageViewer
except Exception as _viewer_exc:  # noqa: BLE001 - pragma: no cover
    _VIEWER_IMPORT_ERROR = f"{type(_viewer_exc).__name__}: {_viewer_exc}"
    ImageViewer = None  # type: ignore[assignment]

#: The sci-fi HUD overlay (Tkinter). Optional and self-disabling; a missing
#: display or tk never touches the voice client.
_OVERLAY_IMPORT_ERROR: Optional[str] = None
try:
    from client.overlay import OverlayHUD
except Exception as _overlay_exc:  # noqa: BLE001 - pragma: no cover
    _OVERLAY_IMPORT_ERROR = f"{type(_overlay_exc).__name__}: {_overlay_exc}"
    OverlayHUD = None  # type: ignore[assignment]

#: Audio format announced in ``utterance_start`` (SPEC §4).
PCM_FORMAT = getattr(_protocol, "AUDIO_FORMAT", "pcm_s16le")
CHANNELS = int(getattr(_protocol, "AUDIO_CHANNELS", 1))
#: How long we wait for a batch of actions before moving on. Must cover the
#: slowest action: ``run_command`` runs PowerShell for up to 30 s (SPEC §8) and
#: BLE devices can take a while too.
ACTION_TIMEOUT_S = 60.0
#: Upper bound for ``action_result.output`` (SPEC §4, C->S #5).
MAX_OUTPUT_CHARS = 4000
#: Short "nothing heard" chirp after a false wake-word trigger.
NO_SPEECH_BEEP_FREQ_HZ = 440.0
NO_SPEECH_BEEP_MS = 90
#: Pause before the follow-up window opens: the room is still ringing with the
#: tail of our own reply (speaker-to-mic echo), which VAD would otherwise pick
#: up as speech and send to the server as a phantom empty utterance.
#: Two seconds at the owner's request - half a second was not enough room on a
#: TV with real speakers, and it also gives the person a beat to start talking
#: instead of the microphone opening while the reply is still hanging in the air.
FOLLOWUP_ECHO_GUARD_S = 2.0
#: "Thinking" sounds: when the server takes longer than this to start replying
#: (vision, tool rounds), a soft two-tone blip repeats so the user knows Jarvis
#: is working rather than stuck. Silenced the moment the reply begins.
THINKING_DELAY_S = 1.5
THINKING_INTERVAL_S = 2.2
THINKING_BLIP_FREQS_HZ = (520.0, 660.0)
THINKING_BLIP_MS = 60
THINKING_VOLUME = 0.14
#: Mic audio recorded while the ack beep was playing: everything captured in the
#: last ``BEEP_MS + BEEP_ECHO_GUARD_MS`` is our own beep (tone + output latency)
#: and is dropped; anything older is the user already speaking and is kept, so a
#: command said in one breath ("rowan, turn on the light") is not cut off.
BEEP_ECHO_GUARD_MS = 250

#: After a proactive message (a greeting that asks a question) the client
#: listens this long WITHOUT the wake word, so the person can just answer.
PROACTIVE_LISTEN_S = 8.0

#: How long to wait after hiding the HUD before grabbing the screen. The window
#: is gone the moment Qt processes the hide, but the desktop compositor still
#: has to repaint what was underneath it; without this pause the screenshot can
#: catch the overlay mid-fade and the assistant ends up describing its own glow.
OVERLAY_SETTLE_S = 0.12

RESULT_OK = "ok"
RESULT_NO_SPEECH = "no_speech"
RESULT_ERROR = "error"
#: The user said the wake word while the reply was playing: playback was cut
#: and the client goes straight back to listening.
RESULT_BARGE_IN = "barge_in"
RESULT_DISMISSED = "dismissed"

#: Routing modes of the reader task (v1.4). The conversation loop owns the flag.
#: ``idle`` — proactive audio is played and camera/screen requests answered;
#: ``conversation`` — every message belongs to the utterance in flight and is
#: buffered for :meth:`JarvisClient._receive_response`.
MODE_IDLE = "idle"
MODE_CONVERSATION = "conversation"
#: How often a waiting conversation re-checks that the reader is still alive.
INBOX_POLL_S = 0.5


class _LinkDown:
    """Sentinel put into the inbox when the reader task ends."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<link down>"


#: Singleton sentinel: waking a waiting conversation when the socket dies.
LINK_DOWN = _LinkDown()


class _NoOverlay:
    """Stand-in when the overlay module cannot be imported: every call is a no-op."""

    enabled = False

    def suspend_capture(self, timeout=2.0):
        return True

    def confirm_voice(self, name, callback):
        callback(False)

    def __getattr__(self, _name: str):  # pragma: no cover - trivial
        def _noop(*_a, **_k):
            return None
        return _noop


class _Stopping(Exception):
    """Internal: Ctrl+C was pressed, unwind the audio loops."""


class _Dismissed(Exception):
    """The user requested silence, with no follow-up recording."""


class _BargedIn(Exception):
    """Internal: the owner said the wake word, abandon whatever we are waiting for.

    Raised from inside the inbox poll rather than checked between messages.
    Waiting for the NEXT message is exactly where a wedged turn parks, and the
    old code only looked at the barge-in flag after a message had arrived — so
    a server that went quiet swallowed the wake word for the whole 420 s
    receive timeout, which is precisely what "I say rowan and he does not
    care" looked like from the room.
    """


def _attr(obj: Any, name: str) -> Any:
    """Read ``name`` from a pydantic model / dataclass / mapping."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clip_output(value: Any) -> Optional[str]:
    """Normalise the dispatcher's ``output`` for ``action_result`` (SPEC §4).

    ``None``/empty stays ``None``; anything longer than :data:`MAX_OUTPUT_CHARS`
    is cut so one chatty command cannot flood the LLM context.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if not text:
        return None
    if len(text) > MAX_OUTPUT_CHARS:
        suffix = "... (truncated)"
        text = text[: max(0, MAX_OUTPUT_CHARS - len(suffix))] + suffix
    return text


def build_hello(cfg_client: Any) -> Dict[str, Any]:
    """Build the ``hello`` payload from the client config (SPEC §4.1)."""
    devices: List[Dict[str, Any]] = []
    for dev in (_attr(cfg_client, "devices") or []):
        name = _opt_str(_attr(dev, "name"))
        if not name:
            continue
        devices.append(
            {
                "name": name,
                "type": str(_attr(dev, "type") or ""),
                "area": _opt_str(_attr(dev, "area")),
                "description": _opt_str(_attr(dev, "description")),
            }
        )
    return {
        "type": MSG_HELLO,
        "client_id": str(_attr(cfg_client, "client_id") or "client"),
        "workplace_name": str(_attr(cfg_client, 'workplace_name') or _attr(cfg_client, 'client_id') or 'client'),
        "camera_name": str(_attr(_attr(cfg_client, 'camera'), 'name') or 'Основная камера'),
        "capabilities": ["voice_confirmation", "live_transcript", _protocol.CAP_CAMERA_CLIP],
        "devices": devices,
    }


def resolve_path(raw: Any) -> Path:
    """Resolve a config path relative to the current dir, then to the repo root."""
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path):
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


class JarvisClient:
    """Wires audio, wake word, VAD, WebSocket, screen capture and actions together."""

    def __init__(self, cfg: Any) -> None:
        self._stopping = False
        self.cfg = cfg
        self.ccfg = cfg.client

        audio_cfg = self.ccfg.audio
        self.sample_rate = int(audio_cfg.sample_rate)
        self.frame_ms = FRAME_MS
        self.frame_bytes = frame_bytes(self.sample_rate, self.frame_ms)
        processor = None
        if getattr(audio_cfg, 'echo_cancellation', False) or getattr(audio_cfg, 'noise_suppression', False):
            from client.audio_processing import AudioPreprocessor
            processor = AudioPreprocessor(
                self.sample_rate, echo_cancellation=getattr(audio_cfg, 'echo_cancellation', False),
                noise_suppression=getattr(audio_cfg, 'noise_suppression', False),
                ns_level=getattr(audio_cfg, 'noise_suppression_level', 1),
                output_device=audio_cfg.output_device,
            )
        self.audio_in = AudioInput(
            device=audio_cfg.input_device,
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            processor=processor,
        )
        self.audio_out = AudioOutput(
            device=audio_cfg.output_device,
            default_sample_rate=self.sample_rate,
        )

        vad_cfg = self.ccfg.vad
        pre_roll_ms = max(0, int(vad_cfg.pre_roll_ms))
        self.vad = VadRecorder(
            aggressiveness=int(vad_cfg.aggressiveness),
            silence_ms=int(vad_cfg.silence_ms),
            max_utterance_s=float(vad_cfg.max_utterance_s),
            sample_rate=self.sample_rate,
            frame_ms=self.frame_ms,
            pre_roll_ms=pre_roll_ms,
            min_speech_ms=int(getattr(vad_cfg, "min_speech_ms", 250)),
            energy_endpoint=True,
        )
        self.preroll = RingBuffer(int(math.ceil(max(2200, pre_roll_ms) / float(self.frame_ms))))

        self.followup_window_s = float(_attr(self.ccfg, "followup_window_s") or 0.0)
        self.attention_mode = str(_attr(self.ccfg, "attention_mode") or "wake_word")
        self._last_room_speech_notice = 0.0
        raw_thinking = _attr(self.ccfg, "thinking_sounds")
        self.thinking_sounds = True if raw_thinking is None else bool(raw_thinking)
        self._thinking_task: Optional[asyncio.Task] = None
        self._barge_task: Optional[asyncio.Task] = None
        self._barged = False
        self._interrupt_id = ''
        self._notice_tts = False
        self._enrollment_until = 0.0
        self._live_turn_id = ''
        self._recording_live = False
        self._selection_until = 0.0
        self._quiet_turn = False
        self._dismiss_task = None
        self._dismiss_id = ''
        self._dismiss_ack = asyncio.Event()
        self.silence = None
        self._wake_muted = False
        self._silence_muted = False
        self.enrollment_vad = VadRecorder(aggressiveness=int(vad_cfg.aggressiveness), silence_ms=4000,
                                          max_utterance_s=45, sample_rate=self.sample_rate,
                                          pre_roll_ms=pre_roll_ms, min_speech_ms=250)

        self.registry = build_registry(self.ccfg)
        self.dispatcher = Dispatcher(self.ccfg, self.registry)
        self.ws = WSClient(
            url=str(self.ccfg.server_url),
            hello=build_hello(self.ccfg),
            should_stop=lambda: self._stopping,
        )

        self.wake: Optional[ConfirmedWakeDetector] = None
        self._started = False
        self._action_task: Optional[asyncio.Task] = None
        self._tts_active = False
        self._tts_bytes = 0
        self._last_say = ""
        #: Server's follow-up window request from the last reply (say.listen_s).
        self._listen_hint_s = 0.0
        #: v1.7: HUD caption the last reply asked for (say.status), shown for
        #: the follow-up window it opens - voice enrollment progress.
        self._say_status = ""
        #: v1.7: bumped on every caption, so an old caption's timer cannot
        #: clear a newer one.
        self._status_seq = 0
        #: True while a background caption (MSG_STATUS) owns the HUD state.
        self._status_owns_hud = False
        #: Set when a proactive message finished playing: answer without wake word.
        self._proactive_listen_s = 0.0
        #: Wake word spellings, lowercased - to spot our own name in reply text.
        wake_cfg = self.ccfg.wakeword
        self._wake_phrases = [
            str(p).strip().lower()
            for p in ([_attr(wake_cfg, "word")] + list(_attr(wake_cfg, "phrases") or []))
            if str(p or "").strip()
        ]

        # -- v1.4: one reader task owns the socket -----------------------
        #: Messages belonging to the utterance in flight (text and binary).
        self._inbox: "asyncio.Queue[Any]" = asyncio.Queue()
        self._mode = MODE_IDLE
        self._reader_task: Optional[asyncio.Task] = None
        self._camera_clip_task: Optional[asyncio.Task] = None
        #: Held around every header+binary pair we send, and for the whole
        #: duration of a streamed utterance: the server routes incoming binary
        #: frames by the header that announced them, so a camera JPEG must never
        #: interleave with microphone audio or with a screenshot.
        self._wire_lock = asyncio.Lock()
        # -- proactive (unprompted) playback state, owned by the reader ---
        self._idle_stream_active = False   # between a proactive tts_start/tts_end
        self._idle_tts_active = False      # ...and the speaker is accepting it
        self._idle_tts_bytes = 0
        self._idle_interrupted = False     # the wake word cut the greeting
        self._idle_playing = False         # audio still queued for the speaker
        self._idle_drain_task: Optional[asyncio.Task] = None

        camera_cfg = _attr(self.ccfg, "camera")
        self.camera: Optional[Any] = None
        if CameraService is None:
            log.debug("Camera support is not importable: %s", _CAMERA_IMPORT_ERROR)
        elif camera_cfg is None:
            log.debug("No client.camera section in the config - running voice only")
        else:
            self.camera = CameraService(camera_cfg)

        # -- v1.6: detections photo (find_object's image_show) -------------
        self.viewer: Optional[Any] = ImageViewer() if ImageViewer is not None else None
        if self.viewer is None:
            log.debug("The detections viewer is not importable: %s", _VIEWER_IMPORT_ERROR)

        # -- sci-fi HUD overlay --------------------------------------------
        if OverlayHUD is not None:
            self.overlay: Any = OverlayHUD(_attr(self.ccfg, "overlay"))
        else:
            log.debug("The overlay HUD is not importable: %s", _OVERLAY_IMPORT_ERROR)
            self.overlay = _NoOverlay()
        #: The image_show header awaiting its single binary JPEG frame.
        self._pending_image_show: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # setup / teardown
    # ------------------------------------------------------------------
    def _install_signal_handler(self) -> None:
        def handler(signum, frame):  # noqa: ANN001
            if self._stopping:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                raise KeyboardInterrupt
            self._stopping = True
            log.info("Ctrl+C received - shutting down...")

        try:
            signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError) as exc:  # pragma: no cover - non-main thread
            log.debug("Could not install the SIGINT handler: %s", exc)

    async def _setup_wakeword(self) -> None:
        wake_cfg = self.ccfg.wakeword
        phrases = [p for p in (list(_attr(wake_cfg, "phrases") or [])) if str(p).strip()]
        if not phrases:
            phrases = [str(wake_cfg.word)]
        model_path = resolve_path(wake_cfg.vosk_model)
        wake = await asyncio.to_thread(
            WakeWordDetector, model_path, phrases, self.sample_rate
        )
        self.silence = await asyncio.to_thread(
            SilenceDetector, wake, REPO_ROOT / 'models' / 'vosk-model-small-ru-0.22'
        )
        self.wake = await asyncio.to_thread(ConfirmedWakeDetector, wake)

    async def _shutdown(self) -> None:
        self._stopping = True
        await self._stop_reader()
        await self._cancel_task(self._dismiss_task, 'silence acknowledgement')
        await self._cancel_task(self._idle_drain_task, "proactive playback")
        self._idle_drain_task = None
        camera = self.camera
        if camera is not None:
            try:
                await asyncio.to_thread(camera.stop)
            except Exception as exc:  # pragma: no cover - teardown
                log.debug("Error while stopping the camera: %s", exc)
        viewer = self.viewer
        if viewer is not None:
            try:
                await asyncio.to_thread(viewer.close)
            except Exception as exc:  # pragma: no cover - teardown
                log.debug("Error while stopping the detections viewer: %s", exc)
        try:
            self.overlay.stop()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while stopping the overlay: %s", exc)
        task, self._action_task = self._action_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - teardown
                log.debug("Error while finishing actions: %s", exc)
        try:
            await self.dispatcher.browser.close()
        except Exception as exc:
            log.debug('Error closing the Rowan browser: %s', exc)
        try:
            self.audio_in.close()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the microphone: %s", exc)
        try:
            await self.audio_out.aclose()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the audio output: %s", exc)
        try:
            await self.ws.close()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the connection: %s", exc)
        if self._started:
            log.info("Client stopped")

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task], what: str) -> None:
        """Cancel a helper task and swallow whatever it ends with."""
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - helpers must not break teardown
            log.debug("The %s task ended with: %s", what, exc)

    def _start_camera(self) -> None:
        """Start the optional camera service (SPEC v1.4); never fatal."""
        camera = self.camera
        if camera is None:
            if _CAMERA_IMPORT_ERROR is not None:
                log.warning(
                    "Camera support is unavailable (%s) - running voice only",
                    _CAMERA_IMPORT_ERROR,
                )
            return
        try:
            camera.start(
                asyncio.get_running_loop(),
                self.ws.send_json,
                self.ws.send_bytes,
                send_lock=self._wire_lock,
            )
        except Exception as exc:  # noqa: BLE001 - the camera is never worth a crash
            log.warning("Could not start the camera service: %s - running voice only", exc)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self._install_signal_handler()
        try:
            await self._setup_wakeword()
            self.audio_in.start()
            self._start_camera()
            try:
                self.overlay.start()
            except Exception as exc:  # noqa: BLE001 - the HUD is never worth a crash
                log.debug("Could not start the overlay HUD: %s", exc)
            self._started = True
            log.info(
                "Jarvis client started: client_id=%s, server=%s",
                _attr(self.ccfg, "client_id"),
                self.ccfg.server_url,
            )
            while not self._stopping:
                try:
                    await self._ensure_link()
                    await self._conversation()
                except _Stopping:
                    break
                except WSDisconnected as exc:
                    if self._stopping:
                        break
                    log.warning("Lost the connection to the server: %s", exc)
                    await self._beep(ERROR_BEEP_FREQ_HZ, ERROR_BEEP_MS)
        finally:
            await self._shutdown()

    # ------------------------------------------------------------------
    # v1.4: the connection and its single reader task
    # ------------------------------------------------------------------
    def _reader_alive(self) -> bool:
        task = self._reader_task
        return task is not None and not task.done()

    async def _ensure_link(self) -> None:
        """Guarantee a live connection with exactly ONE task reading it.

        The handshake in ``ws.ensure_connected`` consumes messages itself, so
        the reader is always stopped first: two consumers on one socket would
        each swallow half of the reply.
        """
        if self.ws.connected and self._reader_alive():
            return
        await self._stop_reader()
        await self.ws.ensure_connected()
        self._start_reader()

    def _start_reader(self) -> None:
        if self._reader_alive():
            return
        self._drain_inbox("stale message")
        self._reader_task = asyncio.get_running_loop().create_task(
            self._reader_loop(), name="jarvis-ws-reader"
        )
        log.debug("The socket reader task is running")

    async def _stop_reader(self) -> None:
        self._recording_live = False
        task, self._reader_task = self._reader_task, None
        await self._cancel_task(task, "socket reader")
        clip, self._camera_clip_task = getattr(self, '_camera_clip_task', None), None
        await self._cancel_task(clip, 'camera clip')
        self._idle_stream_active = False
        self._idle_tts_active = False
        # v1.6: an image_show header with no binary yet must not survive a
        # dropped connection - the next frame on a fresh connection would
        # otherwise be misrouted to the viewer instead of audio/TTS.
        self._pending_image_show = None
        self._drain_inbox("message from the previous connection")

    def _drain_inbox(self, what: str) -> int:
        """Throw away buffered messages that can no longer belong to a reply."""
        dropped = 0
        while True:
            try:
                item = self._inbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not LINK_DOWN:
                dropped += 1
        if dropped:
            log.debug("Discarded %d %s(s)", dropped, what)
        return dropped

    async def _reader_loop(self) -> None:
        """Own ``ws.recv()`` for the lifetime of this connection (SPEC v1.4)."""
        try:
            while not self._stopping:
                msg = await self.ws.recv(timeout=WAIT_FOREVER)
                await self._route_message(msg)
        except asyncio.CancelledError:
            raise
        except WSDisconnected as exc:
            if not self._stopping:
                log.info("The socket reader stopped: %s", exc)
        except Exception as exc:  # noqa: BLE001 - one bad message must not deafen us
            log.error("The socket reader crashed: %s", exc)
            log.debug("Details:", exc_info=True)
            self.ws.drop()
        finally:
            self._idle_stream_active = False
            self._idle_tts_active = False
            try:
                self._inbox.put_nowait(LINK_DOWN)
            except Exception:  # pragma: no cover - an unbounded queue cannot fail
                pass

    async def _route_message(self, msg: Any) -> None:
        """Send one server message where it belongs (see the module docstring)."""
        if isinstance(msg, dict) and msg.get('type') == _protocol.MSG_CAMERA_CLIP_REQUEST:
            # A 3–10 s recording must never occupy the sole socket reader.
            task = getattr(self, '_camera_clip_task', None)
            if task is not None and not task.done():
                await self.ws.send_json({'type': _protocol.MSG_CAMERA_CLIP_ERROR,
                    'id': str(msg.get('id') or '')[:100], 'error': 'A camera clip is already in progress.'})
            else:
                self._camera_clip_task = asyncio.create_task(self._handle_camera_clip_request(msg))
            return
        if isinstance(msg, dict) and msg.get('type') == _protocol.MSG_DISMISSED:
            request_id = str(msg.get('id') or '')
            if not request_id or request_id == self._dismiss_id:
                self._silence_locally()
                self._dismiss_ack.set()
            return
        if getattr(self, '_quiet_turn', False):
            # Camera presence may continue; no stale speech, action or overlay
            # update is allowed to revive a dismissed conversation.
            if isinstance(msg, dict) and msg.get('type') == MSG_CAMERA_REQUEST:
                await self._handle_camera_request(msg)
            return
        if isinstance(msg, bytes):
            if self._pending_image_show is not None:
                # v1.6: the JPEG announced by an image_show header, in EITHER
                # mode - never microphone audio, TTS or a conversation message.
                await self._on_image_show_binary(msg)
            elif self._idle_stream_active:
                await self._on_idle_tts_chunk(msg)
            elif self._mode == MODE_CONVERSATION:
                self._inbox.put_nowait(msg)
            else:
                log.debug("Binary frame outside of any stream (%d bytes) - ignored", len(msg))
            return

        mtype = msg.get("type")
        if mtype == _protocol.MSG_TRANSCRIPT_PARTIAL:
            if (getattr(self, '_recording_live', False)
                    and msg.get('utterance_id') == self._live_turn_id):
                self.overlay.transcript(msg)
            return
        if mtype == MSG_IMAGE_SHOW:
            # v1.6: handled directly here, in both idle and conversation mode -
            # the header just announces the ONE binary frame that follows it.
            if msg.get("hide"):
                # No frame follows: the owner asked to dismiss the photo.
                viewer = self.viewer
                if viewer is not None:
                    try:
                        viewer.hide()
                    except Exception as exc:  # noqa: BLE001 - never fatal
                        log.debug("Could not hide the photo: %s", exc)
                return
            self._pending_image_show = dict(msg)
            return
        if mtype == MSG_CAMERA_REQUEST:
            # Answered in both modes: the server pulls frames for
            # look_at_camera / enroll_face whenever it likes.
            await self._handle_camera_request(msg)
            return
        if self._idle_stream_active:
            # The tail of a proactive block stays with the reader even if a
            # conversation started meanwhile (the wake word interrupted it):
            # its binary frames are NOT the reply the conversation waits for.
            if mtype == MSG_TTS_END:
                await self._on_idle_tts_end()
                return
            if mtype == MSG_TTS_START:  # pragma: no cover - defensive
                log.debug("A new TTS stream began before the proactive one ended")
                await self._on_idle_tts_end()
        if self._mode == MODE_CONVERSATION:
            self._inbox.put_nowait(msg)
            return
        await self._handle_idle_message(mtype, msg)

    async def _handle_idle_message(self, mtype: Any, msg: Dict[str, Any]) -> None:
        """Handle a message that arrived between utterances (SPEC v1.4)."""
        if mtype == _protocol.MSG_NOTICE:
            self._interrupt_id = str(msg.get('id') or '')
            self._show_status(str(msg.get('text') or ''), 45)
        elif mtype == _protocol.MSG_CHAT:
            self.overlay.chat(msg)
        elif mtype == MSG_SAY:
            self._last_say = str(msg.get("text") or "").strip()
            log.info("Unprompted message: %s", self._last_say or "(empty)")
        elif mtype == MSG_TTS_START:
            await self._on_idle_tts_start(msg)
        elif mtype == MSG_TTS_END:
            log.debug("tts_end without a proactive stream - ignored")
        elif mtype == MSG_SCREENSHOT_REQUEST:
            await self._handle_screenshot_request(msg)
        elif mtype == MSG_ACTIONS:
            log.info("Actions received between utterances")
            await self._start_actions(msg.get("items") or [])
        elif mtype == MSG_TRANSCRIPT:
            log.debug("Transcript outside of a conversation: %s", msg.get("text"))
        elif mtype == MSG_ERROR:
            log.error("Server error between utterances: %s", msg.get("message"))
        elif mtype == MSG_READY:
            log.debug("The server sent ready")
        elif mtype == MSG_STATUS:
            self._on_status_message(msg, in_conversation=False)
        elif mtype == MSG_SPEAKER:
            self._on_speaker_message(msg)
        else:
            log.warning("Unknown message type from the server: %r", mtype)

    # -- HUD captions (v1.7) --------------------------------------------------

    def _show_status(self, text: str, ttl_s: float) -> None:
        """Put ``text`` on the HUD for ``ttl_s`` seconds (empty clears it)."""
        self._status_seq += 1
        seq = self._status_seq
        self.overlay.set_status(text or "")
        if text and ttl_s > 0:
            asyncio.get_running_loop().call_later(ttl_s, self._expire_status, seq)

    def _expire_status(self, seq: int) -> None:
        if seq != self._status_seq:
            return  # a newer caption replaced this one
        self.overlay.set_status("")
        if self._status_owns_hud:
            self._status_owns_hud = False
            if self._mode == MODE_IDLE:
                self.overlay.set_state("idle")

    def _on_speaker_message(self, msg: Dict[str, Any]) -> None:
        """MSG_SPEAKER: put the recognised person's name in the middle of the TV.

        Cleared when the turn ends, so a name is only ever on screen while it
        is actually about the person talking right now.
        """
        name = str(msg.get("name") or "").strip()
        try:
            score = float(msg.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        log.info("Recognised speaker: %s (%.2f)", name or "(nobody)", score)
        self.overlay.speaker(name)

    def _on_status_message(self, msg: Dict[str, Any], in_conversation: bool) -> None:
        """MSG_STATUS: a caption for background work, e.g. face enrollment photos.

        Between turns the HUD is normally dark, so a caption alone would float
        over an invisible orb; the working state is lit behind it until the
        caption expires. During a conversation the turn owns the HUD state and
        only the caption changes.
        """
        text = str(msg.get("text") or "").strip()
        try:
            ttl_s = float(msg.get("ttl_s") or _protocol.DEFAULT_STATUS_TTL_S)
        except (TypeError, ValueError):
            ttl_s = _protocol.DEFAULT_STATUS_TTL_S
        log.info("Status: %s", text or "(cleared)")
        if not in_conversation and self._mode == MODE_IDLE:
            if text:
                self._status_owns_hud = True
                self.overlay.set_state("thinking")
            elif self._status_owns_hud:
                self._status_owns_hud = False
                self.overlay.set_state("idle")
        self._show_status(text, ttl_s)

    # -- proactive playback (greetings): only ever while idle ----------------

    async def _on_idle_tts_start(self, msg: Dict[str, Any]) -> None:
        sample_rate = int(msg.get("sr") or self.sample_rate)
        fmt = str(msg.get("format") or PCM_FORMAT)
        channels = int(msg.get("channels") or CHANNELS)
        if fmt != PCM_FORMAT or channels != CHANNELS:
            log.warning(
                "The server announced an unexpected TTS format: %s, %d channel(s) - playing it as %s mono",
                fmt, channels, PCM_FORMAT,
            )
        self._idle_tts_bytes = 0
        self._idle_interrupted = False
        # Set before opening the device: even if playback fails, the frames of
        # this block belong to the reader and must not reach the inbox.
        self._idle_stream_active = True
        try:
            await self.audio_out.open(sample_rate)
        except Exception as exc:  # noqa: BLE001 - audio device issues
            log.error("Could not open the audio output at %d Hz: %s", sample_rate, exc)
            self._idle_tts_active = False
            return
        self._idle_tts_active = True
        if getattr(self, '_quiet_turn', False):
            self._idle_tts_active = self._idle_stream_active = False
            return
        self._idle_playing = True
        self.overlay.set_state("speaking")
        log.info("Playing an unprompted message (%d Hz)", sample_rate)

    async def _on_idle_tts_chunk(self, data: bytes) -> None:
        if not self._idle_tts_active or self._idle_interrupted:
            return  # interrupted or muted: swallow the rest of the block
        self._idle_tts_bytes += len(data)
        await self.audio_out.write(data)

    async def _on_idle_tts_end(self) -> None:
        self._idle_stream_active = False
        self._idle_tts_active = False
        self.overlay.set_state("idle")
        if self._idle_interrupted:
            log.debug("The unprompted message was cut after %d bytes", self._idle_tts_bytes)
        elif self._idle_tts_bytes:
            log.debug("Played %d bytes of the unprompted message", self._idle_tts_bytes)
        else:
            log.info("The unprompted message carried no audio")
        # Waiting for the speaker to go quiet must not stop the reader from
        # reading, so the drain runs in its own task.
        if self._idle_drain_task is None or self._idle_drain_task.done():
            self._idle_drain_task = asyncio.get_running_loop().create_task(
                self._finish_idle_playback(), name="jarvis-proactive-drain"
            )

    async def _finish_idle_playback(self) -> None:
        try:
            await self.audio_out.drain()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Error while finishing the unprompted playback: %s", exc)
        finally:
            if not self._idle_stream_active:
                interrupted = self._idle_interrupted
                self._idle_playing = False
                self._idle_interrupted = False
                if not interrupted and self._idle_tts_bytes and self.attention_mode == "window":
                    # The greeting asked a question: give the person a window to
                    # simply answer instead of demanding the wake word first.
                    self._proactive_listen_s = PROACTIVE_LISTEN_S
                    log.info(
                        "Proactive message finished - listening for an answer "
                        "for %.0f s without the wake word", PROACTIVE_LISTEN_S,
                    )

    def _interrupt_idle_playback(self) -> bool:
        """Cut an unprompted message short; ``True`` if there was one.

        Called when the wake word fires and when a new utterance starts. The
        ``_idle_stream_active`` flag stays on so the rest of the block (binary
        frames and its ``tts_end``) is still swallowed by the reader.
        """
        if not (self._idle_stream_active or self._idle_tts_active or self._idle_playing):
            return False
        already = self._idle_interrupted
        self._idle_interrupted = True
        self._idle_tts_active = False
        dropped = self.audio_out.cancel_pending()
        if not already:
            log.info("Interrupting the unprompted message")
        log.debug("Dropped %d queued playback chunk(s)", dropped)
        return True

    # -- conversation mode flag (owned by the conversation loop) -------------

    def _enter_conversation(self) -> None:
        """From now on every server message belongs to the utterance in flight."""
        self._interrupt_idle_playback()
        self._drain_inbox("stale message")
        self._mode = MODE_CONVERSATION

    def _leave_conversation(self) -> None:
        """Back to idle routing; anything left over cannot belong to a reply."""
        self._mode = MODE_IDLE
        self._drain_inbox("leftover reply message")

    async def _next_message(self) -> Any:
        """Next message of the conversation, buffered by the reader task.

        Raises :class:`WSDisconnected` when the link died or the server went
        quiet for longer than the transport's receive timeout — the same
        contract ``ws.recv()`` had before the reader existed.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(getattr(self.ws, "recv_timeout", 420.0))
        while True:
            if getattr(self, '_quiet_turn', False):
                raise _Dismissed()
            try:
                item = await asyncio.wait_for(self._inbox.get(), timeout=INBOX_POLL_S)
            except (asyncio.TimeoutError, TimeoutError):
                if self._stopping:
                    raise _Stopping()
                if self._barged:
                    # Checked HERE, inside the wait, not between messages: a
                    # server that is wedged never sends another message, and
                    # the owner is standing in the room repeating the name.
                    raise _BargedIn()
                if not self._reader_alive() and self._inbox.empty():
                    raise WSDisconnected("the connection was lost while waiting for the reply")
                if loop.time() >= deadline:
                    self.ws.drop()
                    raise WSDisconnected("the server is not responding")
                continue
            if item is LINK_DOWN:
                raise WSDisconnected("the connection was lost while waiting for the reply")
            return item

    async def _conversation(self) -> None:
        """One wake-word trigger plus any follow-up turns."""
        if not await self._wait_for_wakeword():
            return
        proactive = self._proactive_listen_s
        self._proactive_listen_s = 0.0
        pre_roll = self.preroll.snapshot()
        self.preroll.clear()
        await self._beep(BEEP_FREQ_HZ, BEEP_MS)
        pre_roll += self._drain_beep_window()
        lead_in: Optional[float] = proactive if proactive > 0 else None
        verify_wake = proactive <= 0 and self.attention_mode == 'wake_word'
        while not self._stopping:
            result = await self._handle_utterance(pre_roll, lead_in, verify_wake=verify_wake)
            if result == RESULT_BARGE_IN:
                # The wake word cut the turn short: drop whatever the abandoned
                # reply left in the inbox, acknowledge, and listen for the new
                # command right away - no wake word needed. The server cancels
                # its in-flight task when the new utterance arrives.
                stale = 0
                while not self._inbox.empty():
                    try:
                        self._inbox.get_nowait()
                        stale += 1
                    except asyncio.QueueEmpty:
                        break
                if stale:
                    log.debug("Dropped %d stale message(s) of the abandoned turn", stale)
                await self._beep(BEEP_FREQ_HZ, BEEP_MS)
                pre_roll = self._drain_beep_window()
                lead_in = None
                verify_wake = False
                continue
            # SPEC §4: the server may ask for a longer follow-up window via
            # say.listen_s (voice enrollment needs room to keep talking).
            hint = self._listen_hint_s
            self._listen_hint_s = 0.0
            window = followup_seconds(self.attention_mode, self.followup_window_s, hint)
            if result != RESULT_OK or window <= 0:
                self.overlay.set_state("idle")
                return
            pre_roll = b""
            lead_in = window
            verify_wake = False
            await asyncio.sleep(FOLLOWUP_ECHO_GUARD_S)
            self.audio_in.clear()
            self.overlay.set_state("listening")
            log.info(
                "Listening for a follow-up for %.1f s (no wake word needed)...",
                window,
            )

    def _drain_beep_window(self) -> bytes:
        """Return the speech captured while the ack beep played (beep itself dropped).

        The queue holds everything recorded since the wake word fired: first the
        user's next words (the output stream is still opening), then the echo of
        our own tone at the very end. Keep the former, drop the latter — feeding
        the beep to the VAD would make every false trigger look like speech.
        """
        frames: List[bytes] = []
        while True:
            frame = self.audio_in.read_frame_nowait()
            if frame is None:
                break
            frames.append(frame)
        if not frames:
            return b""
        skip = int(math.ceil((BEEP_MS + BEEP_ECHO_GUARD_MS) / float(self.frame_ms)))
        kept = frames[:-skip] if skip < len(frames) else []
        if kept:
            log.debug("Kept %d frame(s) recorded while the beep played", len(kept))
        return b"".join(kept)

    async def _wait_for_wakeword(self) -> bool:
        if self.wake is None:  # pragma: no cover - run() always initialises it
            raise RuntimeError("The wake-word detector is not initialised")
        word = _attr(self.ccfg.wakeword, "word")
        log.info("Waiting for the wake word '%s'...", word)
        self.preroll.clear()
        self.audio_in.clear()
        self.wake.reset()
        if self.silence:
            self.silence.reset()
        silence_active = False
        while not self._stopping:
            if not self.ws.connected or not self._reader_alive():
                log.info("No connection to the server - reconnecting...")
                await self._ensure_link()
                self.audio_in.clear()
                self.preroll.clear()
                self.wake.reset()
                log.info("Waiting for the wake word '%s'...", word)
            if self.attention_mode != "window":
                self._proactive_listen_s = 0.0
            if self._proactive_listen_s > 0:
                log.info("Answer window after a proactive message - no wake word needed")
                self.overlay.set_state("listening")
                self.wake.reset()
                return True
            frame = await self.audio_in.read_frame(timeout=0.5)
            if not frame:
                continue
            self.preroll.push(frame)
            active = self._idle_playing and not self._quiet_turn
            if self.silence:
                if active != silence_active:
                    self.silence.reset()
                    silence_active = active
                if active and not mentions_silence_command(self._last_say) and self.silence.accept_frame(frame):
                    self._request_silence()
                    await self._finish_dismissal()
                    self.preroll.clear()
                    self.wake.reset()
                    continue
            # Only an activity bit leaves the idle client, never ambient audio.
            # Do not stall wake detection behind a camera burst's wire lock.
            now = time.monotonic()
            if (not self._idle_playing and self.vad.is_speech(frame)
                    and now - self._last_room_speech_notice >= 1.0
                    and not self._wire_lock.locked()):
                self._last_room_speech_notice = now
                async with self._wire_lock:
                    await self.ws.send_json({"type": _protocol.MSG_ROOM_SPEECH})
            if self.wake.accept_frame(frame):
                if (
                    self._idle_playing
                    and self._says_wake_word(self._last_say)
                ):
                    # That was Rowan pronouncing its own name through the
                    # speakers (greetings do) - not the user. Ignore it.
                    log.debug("Ignoring the wake word heard in our own speech")
                    self.wake.reset()
                    continue
                log.info("Wake word detected")
                await self._finish_dismissal()
                if self._quiet_turn:
                    self._idle_stream_active = False
                    self._pending_image_show = None
                    self._clear_inbox()
                self._quiet_turn = False
                self._dismiss_id = ''
                self.overlay.flash("wake")
                self.overlay.set_state("listening")
                # While idle this loop is also the barge-in watcher: an
                # unprompted greeting is cut off here and the client goes
                # straight on to record what the user wants (SPEC v1.4).
                self._interrupt_idle_playback()
                self.wake.reset()
                return True
        return False

    # ------------------------------------------------------------------
    # one utterance
    # ------------------------------------------------------------------
    async def _read_frame(self) -> Optional[bytes]:
        if self._stopping:
            raise _Stopping()
        frame = await self.audio_in.read_frame(timeout=0.5)
        if self._stopping:
            raise _Stopping()
        return frame

    def _split_frames(self, chunk: bytes) -> Iterator[bytes]:
        """Split buffered audio into ~30 ms pieces for streaming (SPEC §7 step 4)."""
        size = self.frame_bytes
        for start in range(0, len(chunk), size):
            piece = chunk[start : start + size]
            if piece:
                yield piece

    async def _handle_utterance(
        self, pre_roll: bytes, lead_in_s: Optional[float], *, verify_wake: bool = False
    ) -> str:
        sent_start = False
        holding_wire = False

        async def on_audio(chunk: bytes) -> None:
            nonlocal sent_start, holding_wire
            if not sent_start:
                # Everything binary the server receives between utterance_start
                # and utterance_end is microphone audio (SPEC §4), so the wire is
                # held for the whole stream: a camera frame would corrupt it.
                await self._wire_lock.acquire()
                holding_wire = True
                # ...and from this exact point on every incoming message belongs
                # to this utterance, not to an unprompted greeting.
                self._enter_conversation()
                import uuid
                self._live_turn_id = uuid.uuid4().hex
                self._recording_live = True
                await self.ws.send_json(
                    {
                        "type": MSG_UTTERANCE_START,
                        "utterance_id": self._live_turn_id,
                        "verify_wake": verify_wake,
                        "sr": self.sample_rate,
                        "format": PCM_FORMAT,
                        "channels": CHANNELS,
                    }
                )
                sent_start = True
                log.info("Streaming the utterance to the server...")
            for piece in self._split_frames(chunk):
                await self.ws.send_bytes(piece)

        try:
            try:
                recorder = self.enrollment_vad if time.monotonic() < self._enrollment_until else self.vad
                audio = await recorder.record(
                    self._read_frame,
                    pre_roll=pre_roll,
                    lead_in_s=lead_in_s,
                    on_audio=on_audio,
                )
                if audio is None:
                    # on_audio was never called: nothing was sent, so this was a
                    # false trigger and the mode flag was never flipped.
                    log.info("No speech detected - back to waiting for the wake word")
                    await self._beep(NO_SPEECH_BEEP_FREQ_HZ, NO_SPEECH_BEEP_MS, volume=0.22)
                    return RESULT_NO_SPEECH

                await self.ws.send_json({"type": MSG_UTTERANCE_END})
                log.debug(
                    "Sent %.2f s of audio",
                    len(audio) / float(self.sample_rate * SAMPLE_WIDTH),
                )
            finally:
                self._recording_live = False
                # Released before waiting for the reply: the reply's rounds may
                # need the wire themselves (screenshot), and the camera should
                # get its turn again while the server thinks.
                if holding_wire:
                    holding_wire = False
                    self._wire_lock.release()
            return await self._receive_response()
        finally:
            self._leave_conversation()

    async def _receive_response(self) -> str:
        """Handle every server message for one utterance (SPEC §4).

        ``actions`` and ``screenshot_request`` may arrive several times and in
        any order before the reply, so both are handled inside this loop; it
        ends with ``tts_end`` or ``error``. Since v1.4 the messages come from
        the reader task through :meth:`_next_message` instead of the socket —
        whatever arrived while the microphone was still streaming is already
        waiting in the inbox.
        """
        self._tts_active = False
        self._tts_bytes = 0
        self._barged = False
        self._interrupt_id = ''
        self._wake_muted = self._silence_muted = False
        result = RESULT_OK
        notice_playback_task = None

        async def resume_after_notice():
            # Keep consuming tool requests while the spoken question plays.
            # Only wake detection waits, to avoid hearing our own "Rowan".
            await self.audio_out.drain()
            if getattr(self, '_quiet_turn', False):
                return
            self._notice_tts = False
            self._wake_muted = self._silence_muted = False
            if not self._tts_active:
                self.overlay.set_state('thinking')
                self._start_barge_watch()

        self._start_thinking()
        # The wake word requests confirmation while this receiver continues
        # executing the original task's actions and playing its response.
        self._start_barge_watch()
        try:
            while True:
                if getattr(self, '_quiet_turn', False):
                    result = RESULT_DISMISSED
                    break
                if self._barged:
                    log.info("Turn interrupted by the wake word - abandoning the reply")
                    result = RESULT_BARGE_IN
                    break
                msg = await self._next_message()
                if isinstance(msg, bytes):
                    await self._on_tts_chunk(msg)
                    continue

                mtype = msg.get("type")
                if mtype == MSG_TRANSCRIPT:
                    if msg.get('ignored'):
                        self.overlay.transcript({'ignored': True})
                        log.info('False wake discarded; returning to standby')
                        continue
                    text = str(msg.get("text") or "").strip()
                    language = str(msg.get("language") or "?")
                    log.info("Recognised [%s]: %s", language, text or "(empty)")
                    self.overlay.transcript({'text': text, 'provisional': False})
                    for segment in msg.get("segments") or []:
                        log.info("  %s [%.2f-%.2f]: %s", segment.get("speaker", "Unknown"),
                                 segment.get("start", 0), segment.get("end", 0),
                                 segment.get("text") or "[overlapping / unclear speech]")
                elif mtype == _protocol.MSG_NOTICE:
                    self._interrupt_id = str(msg.get('id') or '')
                    self._silence_muted = mentions_silence_command(msg.get('text') or '')
                    self._show_status(str(msg.get('text') or ''), 45)
                elif mtype == _protocol.MSG_CHAT:
                    self._selection_until = time.monotonic() + 90 if msg.get('selection_active') else 0.0
                    self.overlay.chat(msg)
                elif mtype == MSG_ACTIONS:
                    await self._start_actions(msg.get("items") or [])
                elif mtype == _protocol.MSG_VOICE_CONFIRMATION:
                    await self._handle_voice_confirmation(msg)
                elif mtype == MSG_SCREENSHOT_REQUEST:
                    await self._handle_screenshot_request(msg)
                elif mtype == MSG_SAY:
                    await self._stop_thinking()
                    self._last_say = str(msg.get("text") or "").strip()
                    self._silence_muted = mentions_silence_command(self._last_say)
                    self.overlay.chat_reply(self._last_say, str(msg.get("enrollment_sentence") or ""))
                    self._enrollment_until = time.monotonic() + 180 if msg.get('enrollment_sentence') else 0.0
                    try:
                        self._listen_hint_s = followup_seconds(
                            self.attention_mode, 0, float(msg.get("listen_s") or 0.0)
                        )
                    except (TypeError, ValueError):
                        self._listen_hint_s = 0.0
                    self._say_status = str(
                        msg.get(_protocol.SAY_STATUS_FIELD) or ""
                    ).strip()
                    log.info("Reply: %s", self._last_say or "(empty)")
                elif mtype == MSG_TTS_START:
                    self._notice_tts = msg.get('purpose') == 'notice'
                    await self._stop_thinking()
                    self.overlay.set_state("speaking")
                    await self._on_tts_start(msg)
                    if self._notice_tts or self._says_wake_word(self._last_say):
                        # Rowan is about to SAY its own name: the mic would hear
                        # it from the speakers and barge in on itself. Stop the
                        # watcher for this one playback.
                        log.info("Reply contains the wake word - barge-in off for it")
                        self._wake_muted = True
                elif mtype == MSG_TTS_END:
                    if msg.get('purpose') == 'notice':
                        self._tts_active = False
                        if notice_playback_task is not None:
                            notice_playback_task.cancel()
                        notice_playback_task = asyncio.create_task(resume_after_notice())
                        continue
                    self._tts_active = False
                    await self.audio_out.drain()
                    await self._stop_barge_watch()
                    if getattr(self, '_quiet_turn', False):
                        result = RESULT_DISMISSED
                        break
                    self.overlay.set_state(
                        "listening" if self._listen_hint_s else "idle"
                    )
                    # A caption the reply asked for stays up while the person
                    # answers; otherwise the working caption is cleared.
                    self._show_status(
                        self._say_status,
                        max(self._listen_hint_s, _protocol.DEFAULT_STATUS_TTL_S,
                            self._enrollment_until - time.monotonic(),
                            self._selection_until - time.monotonic())
                        if self._say_status
                        else 0.0,
                    )
                    self._say_status = ""
                    if self._barged:
                        result = RESULT_BARGE_IN
                    elif self._tts_bytes == 0:
                        log.info("The TTS stream was empty - nothing to play")
                    else:
                        log.debug("Played %d bytes of TTS", self._tts_bytes)
                    break
                elif mtype == MSG_ERROR:
                    await self._stop_thinking()
                    log.error("Server error: %s", msg.get("message"))
                    self._tts_active = False
                    await self._beep(ERROR_BEEP_FREQ_HZ, ERROR_BEEP_MS)
                    result = RESULT_ERROR
                    break
                elif mtype == MSG_READY:
                    log.debug("The server sent ready")
                elif mtype == MSG_STATUS:
                    self._on_status_message(msg, in_conversation=True)
                elif mtype == MSG_SPEAKER:
                    self._on_speaker_message(msg)
                else:
                    log.warning("Unknown message type from the server: %r", mtype)
        except _Dismissed:
            result = RESULT_DISMISSED
        except _BargedIn:
            log.info(
                "Turn interrupted by the wake word while waiting on the server "
                "- abandoning the reply"
            )
            self._tts_active = False
            self.audio_out.cancel_pending()
            result = RESULT_BARGE_IN
        finally:
            if notice_playback_task is not None:
                notice_playback_task.cancel()
                try:
                    await notice_playback_task
                except asyncio.CancelledError:
                    pass
            await self._stop_thinking()
            await self._stop_barge_watch()
            await self._finish_dismissal()
            await self._await_actions()
            # The name belongs to the turn that is now over.
            self.overlay.speaker("")
        return result

    def _says_wake_word(self, text: Any) -> bool:
        """True when the given reply text contains one of the wake spellings."""
        lowered = str(text or "").lower()
        return any(phrase in lowered for phrase in self._wake_phrases)

    # -- barge-in: the wake word interrupts playback -------------------------

    async def _barge_loop(self) -> None:
        """Listen for the wake word while the reply is playing."""
        if self.wake is None:
            return
        self.audio_in.clear()  # drop stale audio buffered while the server thought
        self.wake.reset()
        silence = getattr(self, 'silence', None)
        if silence:
            silence.reset()
        preroll = RingBuffer(int(math.ceil(2200 / float(self.frame_ms))))
        while True:
            frame = await self.audio_in.read_frame(timeout=0.5)
            if not frame:
                continue
            preroll.push(frame)
            if silence and not self._silence_muted and silence.accept_frame(frame):
                self._request_silence()
                return
            if getattr(self, '_wake_muted', False):
                continue
            if self.wake.accept_frame(frame):
                log.info("Wake word during work - requesting cancellation confirmation")
                self.wake.reset()
                if self._interrupt_id:
                    token = self._interrupt_id
                    audio = await self.vad.record(self._read_frame, pre_roll=preroll.snapshot(), lead_in_s=5)
                    if audio:
                        async with self._wire_lock:
                            await self.ws.send_json({'type': MSG_UTTERANCE_START, 'sr': self.sample_rate,
                                                     'format': PCM_FORMAT, 'channels': CHANNELS, 'interrupt_id': token})
                            try:
                                for piece in self._split_frames(audio):
                                    await self.ws.send_bytes(piece)
                            finally:
                                await self.ws.send_json({'type': MSG_UTTERANCE_END})
                else:
                    await self.ws.send_json({'type': _protocol.MSG_INTERRUPT_REQUEST})
                preroll.clear()

    def _clear_inbox(self):
        while not self._inbox.empty():
            try:
                self._inbox.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _silence_locally(self):
        self._recording_live = False
        self._quiet_turn = True
        self._interrupt_id = ''
        self._listen_hint_s = self._proactive_listen_s = 0.0
        self._enrollment_until = self._selection_until = 0.0
        self._idle_interrupted = True
        self._idle_playing = self._idle_stream_active = self._idle_tts_active = False
        self._tts_active = False
        self._pending_image_show = None
        self.overlay.cancel_voice_confirmation()
        self.audio_out.cancel_pending()
        if self._action_task is not None:
            self._action_task.cancel()
            self._action_task = None
        self.overlay.typing(False)
        self.overlay.scan_screen(False)
        self.overlay.speaker('')
        self._show_status('', 0)
        self.overlay.chat({})
        self.overlay.set_state('idle')
        self._clear_inbox()

    def _request_silence(self):
        if self._quiet_turn:
            return
        import uuid
        self._silence_locally()
        self._dismiss_id = uuid.uuid4().hex
        self._dismiss_ack.clear()
        # A separate task survives cancellation of the microphone watcher.
        self._dismiss_task = asyncio.create_task(self._send_dismissal(self._dismiss_id))

    async def _send_dismissal(self, request_id):
        try:
            async with self._wire_lock:
                await self.ws.send_json({'type': _protocol.MSG_DISMISS, 'id': request_id})
            await asyncio.wait_for(self._dismiss_ack.wait(), timeout=5)
        except (asyncio.TimeoutError, WSDisconnected):
            # A disconnected server cancels that connection's remaining work.
            self.ws.drop()
        log.info('Dismissed silently; waiting for the wake word')

    async def _finish_dismissal(self):
        task = getattr(self, '_dismiss_task', None)
        if task is not None:
            await asyncio.shield(task)
            self._dismiss_task = None

    def _start_barge_watch(self) -> None:
        if self._barge_task is None:
            self._barge_task = asyncio.get_running_loop().create_task(
                self._barge_loop(), name="jarvis-barge-in"
            )

    async def _stop_barge_watch(self) -> None:
        task, self._barge_task = self._barge_task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - the watcher must never break the loop
            log.debug("Barge-in watcher ended with: %s", exc)

    # -- thinking sounds (the "I'm working on it" blips) ---------------------

    async def _thinking_loop(self) -> None:
        """Soft repeating blips while the server is still working on a reply."""
        await asyncio.sleep(THINKING_DELAY_S)
        while True:
            for freq in THINKING_BLIP_FREQS_HZ:
                if getattr(self, '_quiet_turn', False):
                    return
                await self._beep(freq, THINKING_BLIP_MS, volume=THINKING_VOLUME)
            await asyncio.sleep(THINKING_INTERVAL_S)

    def _start_thinking(self) -> None:
        self.overlay.set_state("thinking")
        if self.thinking_sounds and self._thinking_task is None:
            self._thinking_task = asyncio.get_running_loop().create_task(
                self._thinking_loop(), name="jarvis-thinking-sounds"
            )

    async def _stop_thinking(self) -> None:
        task, self._thinking_task = self._thinking_task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - a sound must never break the loop
            log.debug("Thinking-sound task ended with: %s", exc)

    async def _on_tts_start(self, msg: Dict[str, Any]) -> None:
        sample_rate = int(msg.get("sr") or self.sample_rate)
        fmt = str(msg.get("format") or PCM_FORMAT)
        channels = int(msg.get("channels") or CHANNELS)
        if fmt != PCM_FORMAT or channels != CHANNELS:
            log.warning(
                "The server announced an unexpected TTS format: %s, %d channel(s) - playing it as %s mono",
                fmt, channels, PCM_FORMAT,
            )
        self._tts_bytes = 0
        try:
            await self.audio_out.open(sample_rate)
        except Exception as exc:
            log.error("Could not open the audio output at %d Hz: %s", sample_rate, exc)
            self._tts_active = False
            return
        self._tts_active = True
        log.debug("Receiving TTS: %d Hz", sample_rate)

    async def _on_tts_chunk(self, data: bytes) -> None:
        if not self._tts_active:
            log.debug("Binary frame outside of a TTS stream (%d bytes) - skipping", len(data))
            return
        if self._barged:
            return  # interrupted: swallow the rest of the stream silently
        self._tts_bytes += len(data)
        await self.audio_out.write(data)

    async def _beep(self, freq: float, ms: int, volume: float = 0.35) -> None:
        try:
            await self.audio_out.play_beep(freq, ms, volume)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - audio device issues
            log.warning("Could not play the beep: %s", exc)

    # ------------------------------------------------------------------
    # screen vision (SPEC §4 S->C #4, §7)
    # ------------------------------------------------------------------
    async def _handle_screenshot_request(self, msg: Dict[str, Any]) -> None:
        """Answer ``screenshot_request``: a header plus exactly ONE binary frame.

        The header carries both sizes (SPEC §4, C->S #6): ``w``/``h`` of the
        downscaled image that is actually sent, and ``screen_w``/``screen_h`` of
        the real desktop. The server needs both to turn the vision model's pixel
        coordinates into the normalized ones a ``mouse_click`` action expects.

        A capture failure is reported as ``screenshot_error`` and no binary
        frame is sent, so the server can turn it into a tool result instead of
        waiting for the full 120 s.
        """
        request_id = str(msg.get("id") or "")
        log.info("Screenshot requested (id=%s) - capturing the screen", request_id or "?")
        # The HUD must never be baked into the picture the assistant is about
        # to study, so it goes off screen first and only comes back - shutter,
        # then the working glow in the corner - once the frame is already in
        # hand. OVERLAY_SETTLE_S gives the compositor time to actually drop the
        # window before the screen is grabbed.
        self.overlay.scan_screen(False)
        self.overlay.set_status("")
        try:
            if not await asyncio.to_thread(self.overlay.suspend_capture):
                raise RuntimeError("Overlay did not acknowledge hiding; screenshot cancelled")
            await asyncio.sleep(OVERLAY_SETTLE_S)
            capture: Capture = await asyncio.to_thread(capture_jpeg)
        except asyncio.CancelledError:
            self.overlay.set_status("")
            raise
        except Exception as exc:
            self.overlay.set_status("")
            error = str(exc).strip() or exc.__class__.__name__
            log.error("Screen capture failed: %s", error)
            await self.ws.send_json(
                {"type": MSG_SCREENSHOT_ERROR, "id": request_id, "error": error}
            )
            return
        finally:
            self.overlay.resume_capture()

        if getattr(self, '_quiet_turn', False):
            return
        # Captured: shutter, then the glow returns in the bottom-right corner
        # and sweeps while the server's vision model reads the frame.
        self.overlay.flash("shot")
        self.overlay.set_state("thinking")
        self.overlay.set_status("looking at the screen")
        self.overlay.scan_screen(True)

        # The header and its single binary frame must stay adjacent on the wire.
        async with self._wire_lock:
            await self.ws.send_json(
                {
                    "type": MSG_SCREENSHOT,
                    "id": request_id,
                    "format": SCREENSHOT_FORMAT,
                    "w": capture.w,
                    "h": capture.h,
                    "screen_w": capture.screen_w,
                    "screen_h": capture.screen_h,
                }
            )
            await self.ws.send_bytes(capture.jpeg)
        log.info(
            "Screenshot sent (id=%s): %dx%d image of a %dx%d screen, %d bytes",
            request_id or "?",
            capture.w,
            capture.h,
            capture.screen_w,
            capture.screen_h,
            len(capture.jpeg),
        )
        self.overlay.scan_screen(False)
        self.overlay.set_status("")

    # ------------------------------------------------------------------
    # room camera (SPEC v1.4, S->C camera_request)
    # ------------------------------------------------------------------
    async def _handle_camera_request(self, msg: Dict[str, Any]) -> None:
        """Answer ``camera_request`` with the newest camera frame, in any mode.

        :mod:`client.camera` owns the reply, including ``camera_error`` when it
        has no picture to give (the mirror of ``screenshot_error``). A client
        without a camera answers the error itself so the server's tool call
        fails immediately instead of waiting for its timeout. ``burst`` (v1.4
        burst extension) asks for that many frames instead of one; absent or
        invalid, it defaults to a single frame exactly as before.
        """
        request_id = str(msg.get("id") or "")
        try:
            burst = int(msg.get("burst") or 1)
        except (TypeError, ValueError):
            burst = 1
        full = bool(msg.get("full"))  # v1.6: skip the downscale for this pull
        log.info(
            "Camera frame requested (id=%s)%s%s",
            request_id or "?",
            f" burst={burst}" if burst != 1 else "",
            " full" if full else "",
        )
        camera = self.camera
        if camera is None:
            await self.ws.send_json(
                {
                    "type": MSG_CAMERA_ERROR,
                    "id": request_id,
                    "error": "this client has no camera",
                }
            )
            return
        try:
            await camera.serve_request(request_id, burst, full=full)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the camera never breaks the client
            log.warning("Could not answer the camera request: %s", exc)

    async def _handle_camera_clip_request(self, msg: Dict[str, Any]) -> None:
        try:
            if self.camera is None:
                await self.ws.send_json({'type': _protocol.MSG_CAMERA_CLIP_ERROR,
                    'id': str(msg.get('id') or '')[:100], 'error': 'This client has no camera.'})
                return
            from client.camera_clips import serve_clip
            await serve_clip(self.camera, msg.get('id'), msg.get('seconds', 5), msg.get('fps', 8))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning('Could not answer the camera clip request (%s)', type(exc).__name__)

    # ------------------------------------------------------------------
    # detections photo (SPEC v1.6, S->C image_show)
    # ------------------------------------------------------------------
    async def _on_image_show_binary(self, data: bytes) -> None:
        """Hand the JPEG announced by ``image_show`` to :mod:`client.viewer`.

        Accepted in BOTH idle and conversation mode (see ``_route_message``):
        ``find_object`` can push this mid-utterance, well before the reply's
        own ``say``/``tts_start``. Runs the viewer in a worker thread since
        the first call may lazily import ``cv2``, which must never stall the
        reader task.
        """
        header = self._pending_image_show or {}
        self._pending_image_show = None
        title = str(header.get("title") or "Jarvis")
        if title.startswith('Which person are you?'):
            self.overlay.photo(data)
            return
        try:
            ttl_s = float(header.get("ttl_s") or 0.0)
        except (TypeError, ValueError):
            ttl_s = 0.0
        log.info("Detections photo received (%r, %d bytes)", title, len(data))
        if self.viewer is None:
            log.debug("No detections viewer available - ignoring the photo")
            return
        try:
            await asyncio.to_thread(self.viewer.show, data, title, ttl_s)
        except Exception as exc:  # noqa: BLE001 - never let a viewer bug break the reader
            log.warning("Could not show the detections photo: %s", exc)

    # ------------------------------------------------------------------
    # actions (executed by W3's dispatcher)
    # ------------------------------------------------------------------
    async def _start_actions(self, items: Any) -> None:
        if getattr(self, '_quiet_turn', False):
            return
        if not isinstance(items, list) or not items:
            log.debug("No actions in this batch")
            return
        await self._await_actions()
        if getattr(self, '_quiet_turn', False):
            return
        log.info("Received %d action(s)", len(items))
        self._action_task = asyncio.get_running_loop().create_task(
            self._execute_actions(list(items)), name="jarvis-actions"
        )

    async def _handle_voice_confirmation(self, msg):
        # No automated input may click the confirmation. Finish previous input
        # before showing it; this receiver processes no action batches meanwhile.
        await self._await_actions()
        if getattr(self, '_quiet_turn', False):
            return
        future = asyncio.get_running_loop().create_future()
        loop = asyncio.get_running_loop()
        def resolve(approved):
            if not future.done():
                future.set_result(approved is True)
        def callback(approved):
            if not loop.is_closed():
                loop.call_soon_threadsafe(resolve, approved)
        approved = False
        try:
            details = ({'old_name': str(msg.get('old_name') or '')[:100],
                        'new_name': str(msg.get('new_name') or '')[:100], 'merge': msg.get('merge') is True}
                       if msg.get('kind') == 'rename' else str(msg.get('name') or '')[:100])
            self.overlay.confirm_voice(details, callback)
            approved = await asyncio.wait_for(future, 46)
        except asyncio.TimeoutError:
            pass
        finally:
            self.overlay.cancel_voice_confirmation()
        await self.ws.send_json({'type': _protocol.MSG_VOICE_CONFIRMATION_RESULT,
                                 'id': msg.get('id'), 'approved': approved})

    async def _execute_actions(self, items: List[Any]) -> None:
        reporting = True
        for item in items:
            if getattr(self, '_quiet_turn', False):
                break
            if not isinstance(item, dict):
                log.warning("Skipping a malformed action: %r", item)
                continue
            action_id = str(item.get("id") or "")
            tool = str(item.get("tool") or "")
            args = item.get("args") or {}
            log.info("Executing %s (%s): %s", action_id or "?", tool, {k: '<image omitted>' if k == 'jpeg_base64' else v for k, v in args.items()})
            # HUD: animate the mouse targeting for a click and the typing pulse.
            command = str(args.get("command") or "").strip().lower()
            typing = tool == "pc_control" and command == "type_text"
            if tool == "mouse_click":
                self.overlay.click_at(args.get("x_norm"), args.get("y_norm"))
            if typing:
                self.overlay.typing(True)
            try:
                ok, error, output = await self.dispatcher.execute(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                ok, error, output = False, f"{type(exc).__name__}: {exc}", None
                log.warning("Action %s crashed: %s", action_id or "?", exc)
            if typing:
                self.overlay.typing(False)
            ok = bool(ok)
            output_text = clip_output(output)
            if ok:
                log.info("Action %s done", action_id or "?")
            else:
                log.warning("Action %s failed: %s", action_id or "?", error)
            if output_text:
                log.debug("Action %s output: %s", action_id or "?", output_text[:200])
            if not reporting:
                continue
            try:
                await self.ws.send_json(
                    {
                        "type": MSG_ACTION_RESULT,
                        "id": action_id,
                        "ok": ok,
                        "error": None if ok else (str(error) if error else "unknown error"),
                        "output": output_text,
                    }
                )
            except WSDisconnected as exc:
                # The server waits for these results (SPEC §4) and falls back to
                # a timeout result, but a dead socket must not cancel the actions
                # the user asked for: finish them, stop reporting, and let
                # run()/ensure_connected handle the reconnect.
                log.warning("Could not send the action result: %s", exc)
                reporting = False

    async def _await_actions(self) -> None:
        task = self._action_task
        if task is None:
            return
        if task.done():
            self._action_task = None
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=ACTION_TIMEOUT_S)
            self._action_task = None
        except (asyncio.TimeoutError, TimeoutError):
            log.warning(
                "Actions are taking longer than %.0f s - continuing, will wait later",
                ACTION_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            if getattr(self, '_quiet_turn', False):
                self._action_task = None
                return
            raise
        except Exception as exc:  # pragma: no cover - dispatcher must not raise
            self._action_task = None
            log.warning("Error while executing actions: %s", exc)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="client.main",
        description="Jarvis room client: wake word, VAD, streaming to the brain server",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config.yaml"),
        help="path to config.yaml (default: config.yaml in the repository root)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level: DEBUG, INFO, WARNING, ERROR",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    # Transcripts and app names may contain non-ASCII text that a cp125x/cp866
    # Windows console cannot encode - never let that kill a log call.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - redirected/exotic streams
            pass
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(level=resolved, format=fmt, datefmt="%H:%M:%S")
    # The client normally runs with a hidden console, so the log also goes to a
    # file — that is the only way to debug it after the window went away.
    try:
        log_path = REPO_ROOT / "data" / "client.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
            log_path.unlink()  # crude rotation: start over past 5 MB
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logging.getLogger().addHandler(file_handler)
    except Exception as exc:  # noqa: BLE001 - file logging is best-effort
        logging.getLogger(__name__).warning("File logging unavailable: %s", exc)
    for noisy in ("websockets", "websockets.client", "comtypes", "bleak", "asyncio", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config_path = resolve_path(args.config)
    if not config_path.exists():
        log.error(
            "Config file not found: %s (copy config.example.yaml to config.yaml)",
            config_path,
        )
        return 2

    try:
        cfg = load_config(str(config_path))
        client = JarvisClient(cfg)
    except Exception as exc:
        log.error("Could not start the client: %s", exc)
        log.debug("Details:", exc_info=True)
        return 1

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        log.info("Interrupted by the user")
    except Exception as exc:
        log.error("The client stopped because of an error: %s", exc)
        log.debug("Details:", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
