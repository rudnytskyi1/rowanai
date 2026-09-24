"""WebSocket protocol constants shared by server and client (SPEC section 4).

Control frames are JSON text frames, audio and screenshots are sent as raw binary
frames. Never inline the message-type strings anywhere else -- import them from
here.

Client -> Server
----------------
* ``{"type": MSG_HELLO, "client_id": str, "devices": [...]}``
* ``{"type": MSG_UTTERANCE_START, "sr": 16000, "format": "pcm_s16le", "channels": 1}``
* binary frames: raw PCM s16le mono 16 kHz
* ``{"type": MSG_UTTERANCE_END}``
* ``{"type": MSG_ACTION_RESULT, "id": str, "ok": bool, "error": str | None,
  "output": str | None}``
* ``{"type": MSG_SCREENSHOT, "id": str, "format": "jpeg"}`` followed by exactly ONE
  binary frame with the JPEG bytes, or ``{"type": MSG_SCREENSHOT_ERROR, "id": str,
  "error": str}`` with no binary frame.
* v1.4 camera: ``{"type": MSG_CAMERA_STATE, "persons": int, "objects": {label: count}}``
  -- state only, no video, sent when the picture changes (debounced).
* v1.4 camera: ``{"type": MSG_CAMERA_FRAME, "id": str, "reason": "presence" | "request",
  "w": int, "h": int, "seq": int, "of": int}`` followed by exactly ONE binary frame
  with the JPEG bytes. ``seq``/``of`` (burst extension) are 1-based: several
  ``camera_frame`` header+binary pairs share the same ``id`` -- ``of`` frames total,
  this one is number ``seq`` -- sent back to back under the client's wire lock so
  nothing else interleaves with them. A plain single frame is simply ``seq=1, of=1``.
  ``reason: "presence"`` frames are unsolicited mini-bursts (one burst of
  :data:`CAMERA_BURST_DEFAULT`-or-more frames every ``client.camera.face_check_interval_s``
  while somebody is visible); ``reason: "request"`` answers a :data:`MSG_CAMERA_REQUEST`,
  whose own ``burst`` field says how many frames were asked for. On capture failure:
  ``{"type": MSG_CAMERA_ERROR, "id": str, "error": str}`` with no binary frame (ends
  the whole burst, however many pairs already went out).
* Optional ``camera_clip`` hello capability: ``{type: MSG_CAMERA_CLIP, id,
  format: "mp4", bytes, w, h, seconds, fps}`` followed by one silent MP4 binary
  frame of at most CAMERA_CLIP_MAX_BYTES, or ``{type: MSG_CAMERA_CLIP_ERROR,
  id, error}`` with no binary. Header and MP4 share the client's wire lock.
* ``{"type": MSG_TTS_PREFETCH, "phrases": [{"id": str, "text": str}]}`` (ТЗ 4.8)
  asks for the fixed lines the room must be able to SAY with the hub gone; the
  hub answers with one MSG_TTS_PHRASE header + binary PCM per phrase.

Server -> Client
----------------
* ``{"type": MSG_READY}``
* ``{"type": MSG_TRANSCRIPT, "text": str, "language": str}``
  -- optional ``segments`` contains attributed turns with ``start``, ``end``,
  ``speaker_id``, ``speaker``, ``text``, ``uncertain``; optional ``clarification``
  explains why no addressed command was selected (see SPEC section 4).
* ``{"type": MSG_ACTIONS, "items": [{"id": str, "tool": str, "args": {...}}]}``
  -- may be sent several times per utterance (one per tool round).
* ``{"type": MSG_SCREENSHOT_REQUEST, "id": str}``
* ``{"type": MSG_CAMERA_REQUEST, "id": str, "burst": int, "full": bool}`` -- v1.4:
  pull one or more camera frames. ``burst`` is optional (default
  :data:`CAMERA_BURST_DEFAULT`, capped at :data:`CAMERA_BURST_MAX`); the client
  answers with that many ``camera_frame`` header+binary pairs sharing ``id``
  (see the seq/of note above). ``full`` (v1.6, optional, default false) skips
  the usual downscale for THIS pull only -- used by ``find_object`` so the
  object detector sees the frame at native camera resolution.
* ``{type: MSG_CAMERA_CLIP_REQUEST, id, seconds: 3..10, fps: 5..10}`` is sent
  only when the client advertises ``camera_clip``. Recording uses existing
  camera capture frames; it does not open another camera or run extra YOLO.
* ``{"type": MSG_SOUND_EVENT, "label": str, "conf": 0..1, "at_ms": int}`` --
  v2 (ТЗ F-109): one sound the client's own detector heard (knock, bell,
  alarm, breaking glass, cough). No audio leaves the room; the hub turns the
  label into an alert-rule event (F-702).
* ``{"type": MSG_OBJECT_EVENT, "label": "cat" | "dog" | "package",
  "zone": str, "conf": 0..1, "at_ms": int, "event_id": str}`` -- v2 (ТЗ
  F-311): an object of attention appeared in the room. The client names the
  class by its canonical group and adds the ZONE of the detection (F-309),
  because the room's own frame zones live on the room PC; the hub turns it
  into an alert-rule event that can be limited to that zone.
* ``{"type": MSG_POINT_EVENT, "x": 0..1, "y": 0..1, "at_ms": int,
  "event_id": str}`` -- v2 (ТЗ F-306): the person points at something and the
  client sends WHERE, not WHAT it sees. The hub keeps the point for a few
  seconds and answers "what is this?" about that part of the frame.
* ``{"type": MSG_POSTURE_EVENT, "state": "sleep" | "awake", "at_ms": int,
  "event_id": str}`` -- v2 (ТЗ F-307): the room watched the person's posture
  (YOLO11-pose, one frame in five seconds) and says the person FELL ASLEEP
  lying still in quiet hours, or GOT UP. The frame never leaves the room.
* ``{"type": MSG_COMPUTER_USE_STEP, "id": str, "run_id": str, "ok": bool,
  "index": int, "step": str, "reason": str, "stopped": bool, "at_ms": int}``
  -- v2 (ТЗ F-512): the room reports one computer-use step it ran, refused or
  stopped. ``stopped`` is true when the room itself stopped the run (palm
  gesture or the word «стоп»): the hub closes the run on both ends.
* ``{"type": MSG_OFFLINE_HINT, "reason": str, "eta_s": float}`` (ТЗ 4.8) -- the
  hub is going away on purpose (restart, maintenance), so the client switches
  to its local mode before the socket breaks.
* ``{"type": MSG_TTS_PHRASE, "id": str, "rate": int, "bytes": int}`` (ТЗ 4.8)
  followed by exactly ONE binary frame with the PCM of that fixed line; a line
  the hub could not synthesize comes back with ``error`` and no binary frame.
* ``{"type": MSG_SAY, "text": str, "listen_s": float, "status": str}`` --
  ``listen_s`` and ``status`` are optional. v1.7: ``status`` is a caption the
  client shows on the HUD for the follow-up window this reply opens (voice
  enrollment progress).
* ``{"type": MSG_TTS_START, "sr": int, "format": "pcm_s16le", "channels": 1}``
  followed by binary PCM frames and ``{"type": MSG_TTS_END}``
* ``{"type": MSG_IMAGE_SHOW, "id": str, "w": int, "h": int, "title": str,
  "ttl_s": float}`` followed by exactly ONE binary frame with a JPEG -- v1.6:
  show a photo on the room screen (``find_object`` pushes its annotated
  detections here). ``ttl_s`` is how long the client keeps it up (default 60 on
  the client side if omitted); a newer image replaces whatever is showing.
* ``{"type": MSG_SPEAKER, "name": str, "score": float}`` -- v1.7: the voice was
  recognised; the client shows the name centred on the HUD until the turn ends.
* ``{"type": MSG_STATUS, "text": str, "ttl_s": float}`` -- v1.7: a HUD caption
  for something slow happening in the background (face enrollment photos);
  empty ``text`` clears it. May arrive at any time, nothing is spoken.
* ``{"type": MSG_COMPUTER_USE, "active": bool, "text": str, "run_id": str}``
  -- v2 (ТЗ F-512): the visible «Rowan is in control» badge. While ``active``
  is true the room keeps the overlay badge up; an empty/``false`` message
  clears it at once (the run finished, or the person stopped it).
* ``{"type": MSG_ERROR, "message": str}``

Order per utterance: transcript -> zero or more rounds of actions and/or
screenshot_request (each awaited) -> say -> tts_start ... tts_end.

v1.4: ``say`` + ``tts_start`` ... ``tts_end`` may also arrive UNSOLICITED between
utterances (the proactive greeting of an unknown face), so the client keeps
reading the socket while idle and plays such audio only when it is not busy.

v1.6: ``image_show`` + its binary JPEG may also arrive UNSOLICITED, in EITHER
idle or conversation mode (a tool call mid-utterance pushes it before the
reply's own ``say``/``tts_start``): the reader handles the header+binary pair
itself in both modes, never treating it as microphone audio, TTS or a
conversation message.

Binary-frame disambiguation: the client sends binary frames only between
``utterance_start``/``utterance_end`` and as the single frame announced by a
``screenshot`` or ``camera_frame`` header; they never overlap. A ``camera_frame``
burst is several header+binary pairs sent back to back under the client's wire
lock, so this rule still holds pair by pair. The server's ``image_show`` header
is likewise always followed by exactly one binary frame.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

# --- client -> server -------------------------------------------------------
MSG_HELLO = "hello"
#: Idle local VAD heartbeat, no audio/transcript: defer unsolicited greetings.
MSG_ROOM_SPEECH = "room_speech"
MSG_UTTERANCE_START = "utterance_start"
MSG_UTTERANCE_END = "utterance_end"
#: ТЗ F-711: a client (typically a phone) that ran its own VAD and STT sends
#: the finished words instead of PCM; the hub answers as it would to speech.
MSG_UTTERANCE_TEXT = "utterance_text"
MSG_ACTION_RESULT = "action_result"
#: Reply to a physical confirmation on the room PC; never an LLM tool result.
MSG_VOICE_CONFIRMATION_RESULT = "voice_confirmation_result"
#: v1.1: header announcing the single binary frame with the JPEG screenshot.
MSG_SCREENSHOT = "screenshot"
#: v1.1: the client could not capture the screen; no binary frame follows.
MSG_SCREENSHOT_ERROR = "screenshot_error"
#: v1.4: what the room camera currently sees (people count + object counts).
MSG_CAMERA_STATE = "camera_state"
#: v2 (ТЗ F-109): one sound event the client's own detector heard (knock, bell,
#: alarm, breaking glass, cough). The hub turns it into an alert-rule event
#: (F-702); the client never sends audio for it, only the label and confidence.
MSG_SOUND_EVENT = "sound_event"
#: v2 (ТЗ F-311): один объект внимания (кот, собака, посылка), который
#: появился в кадре, вместе с зоной дома (F-309): ``{"type", "label", "zone",
#: "conf", "at_ms", "event_id"}``. Клиент считает зону сам, потому что
#: ``camera_state`` несёт только счётчики по меткам, а ТЗ хочет «посылка у
#: двери». ``label`` — каноническая группа (``cat``/``dog``/``package``) из
#: ``common.attention_objects``; чужие метки сюда не попадают вовсе.
MSG_OBJECT_EVENT = "object_event"
#: v2 (ТЗ F-306): куда показывает указательный палец, в нормализованных
#: координатах кадра комнаты — ``{"type", "x", "y", "at_ms", "event_id"}``.
#: Точка нужна вопросу «что это?»: человек показывает на предмет, и хаб
#: смотрит именно туда, а не на всю комнату. Кадр при этом остаётся в комнате:
#: уходит только направление.
MSG_POINT_EVENT = "point_event"
#: v2 (ТЗ F-307): комната заметила, что человек уснул или встал —
#: ``{"type", "state": "sleep" | "awake", "at_ms", "event_id"}``. Поза
#: считается на клиенте, кадр остаётся в комнате; хаб по этому событию
#: включает режим сна дома (свет на минимум, беззвучные уведомления) или
#: утреннюю рутину F-420.
MSG_POSTURE_EVENT = "posture_event"
#: v2 (ТЗ F-512): the client reports one computer-use step it executed (or
#: stopped): ``{"type", "id", "run_id", "ok", "index", "step", "reason",
#: "stopped", "at_ms"}``. The hub keeps the trace and, when ``stopped`` is
#: true, closes the run — the person's palm or the word «стоп» must stop the
#: agent on BOTH ends, not only where it was heard.
MSG_COMPUTER_USE_STEP = "computer_use_step"
#: v2 (ТЗ F-512): the «Rowan is in control» badge — ``{"type", "active": bool,
#: "text": str, "run_id": str}``. While ``active`` is true the room shows the
#: overlay badge and keeps it up; the hub clears it the moment the run stops.
MSG_COMPUTER_USE = "computer_use"
#: v2 (ТЗ F-201): the live person TRACKS of one client - each person in the
#: frame keeps its ``track_id`` while it is visible, and a lost track is
#: remembered for up to 30 s so the same person comes back as the same track.
#: ``camera_state`` keeps carrying its count for a client that predates this.
MSG_TRACKS = "tracks"
#: v2 (ТЗ F-202): header announcing ONE binary frame with a person's body
#: crop - JPEG, up to 640 px tall, cut on the client for ReID (F-203).
MSG_BODY_CROP = "body_crop"
#: v1.4: header announcing the single binary frame with a camera JPEG.
MSG_CAMERA_FRAME = "camera_frame"
#: v1.4: the client could not grab a camera frame; no binary frame follows.
MSG_CAMERA_ERROR = "camera_error"
#: A room saying something about its own gear is broken (ТЗ F-702 extension,
#: владелец 2026-09-23: «оно должно постоянно ретраить и в тг увед слать в
#: группу уведов если чет не работает»). Unlike ``camera_error`` this is not an
#: answer to a request: it is the client volunteering that its camera went away
#: or came back, so the hub can tell the owner instead of the room looking dead.
#: ``{"type": MSG_ROOM_HEALTH, "kind": "camera", "ok": false, "detail": "…"}``
MSG_ROOM_HEALTH = "room_health"
MSG_CAMERA_CLIP = "camera_clip"
MSG_CAMERA_CLIP_ERROR = "camera_clip_error"
#: Владелец 2026-09-24: «открыть камеру и чтобы оно показывало видео с камеры на
#: экране и все детекции». Hub -> client: ``{"type": "camera_preview", "on": true,
#: "names": {"<track_id>": "<display name>"}}``. ``on: false`` закрывает окно.
#: The message only turns a local window on and off; no frame of the room leaves
#: the PC because of it (the video is drawn from the client's own capture).
MSG_CAMERA_PREVIEW = "camera_preview"

# --- server -> client -------------------------------------------------------
MSG_READY = "ready"
MSG_TRANSCRIPT = "transcript"
#: Provisional caption during recording; never authorizes actions or history access.
MSG_TRANSCRIPT_PARTIAL = "transcript_partial"
MSG_ACTIONS = "actions"
MSG_VOICE_CONFIRMATION = "voice_confirmation"
#: v1.1: ask the client for a screenshot of the room PC's screen.
MSG_SCREENSHOT_REQUEST = "screenshot_request"
#: v1.4: ask the client for one frame of the room camera.
MSG_CAMERA_REQUEST = "camera_request"
MSG_CAMERA_CLIP_REQUEST = "camera_clip_request"
#: v1.8 (ТЗ F-303): the room's camera goes (or comes) back — the client stops
#: sending frames and shows the indicator; the microphone keeps working, so the
#: same voice can ask for the camera back.
MSG_PRIVACY = "privacy"
#: v2 (ТЗ 4.8): the hub is going away on purpose (restart, maintenance). The
#: client switches to its local mode at once instead of waiting for the socket
#: to break, so the room is told what is happening while the brain is still
#: there to say it.
MSG_OFFLINE_HINT = "offline_hint"
MSG_SAY = "say"
MSG_TTS_START = "tts_start"
MSG_TTS_END = "tts_end"
#: v1.6: show a photo on the room screen (header, then one binary JPEG frame).
MSG_IMAGE_SHOW = "image_show"
#: v2 (ТЗ F-608): play a RECORDING in the room -- ``{"type": MSG_PLAY_AUDIO,
#: "id": str, "rate": int, "seconds": float, "title": str}`` followed by exactly
#: ONE binary frame with raw PCM s16le mono (the same format as TTS). The room
#: hears a real person's voice (the mystery phrase of «угадай, кто сказал»),
#: never synthesized speech: the game is about whose voice it is. ``title`` is
#: an optional HUD caption shown while the clip plays.
MSG_PLAY_AUDIO = "play_audio"
#: v1.7: a short caption for the room screen, ``{"text": str, "ttl_s": float}``,
#: shown on the HUD while something slow happens in the background (face
#: enrollment photos). Empty text clears it. Nothing is spoken.
MSG_STATUS = "status"
#: v2 (ТЗ F-709): a card for the room HUD -- ``{"id": str, "kind": "intercom" |
#: ..., "title": str, "text": str, "ttl_s": float}``. A card is how the room
#: SEES a message that also is (or will be) spoken: the intercom message that
#: waited for its person (F-601) appears on the screen the moment it is said.
#: Like other background frames, a card may be dropped under backpressure -
#: words that matter are also spoken.
MSG_CARD = "card"
#: v2 (ТЗ F-708): what the brain itself is doing, for the room HUD --
#: ``{"state": "online" | "queue" | "offline", "queue": int}``. The hub sends
#: it when a room connects and when the GPU queue picks up or finishes work;
#: ``offline`` is normally the CLIENT's own conclusion when the socket is gone,
#: so the HUD never has to guess whether the room or the brain is the problem.
MSG_HUB_STATUS = "hub_status"
#: v2 (ТЗ 4.8): "synthesize these fixed lines and send me the audio". The
#: client asks for the phrases it must be able to SAY while the hub is gone
#: (there is no TTS on a room PC), and keeps the PCM it gets back.
MSG_TTS_PREFETCH = "tts_prefetch"
#: v2 (ТЗ 4.8): the answer to ``MSG_TTS_PREFETCH`` - one header naming the
#: phrase and its rate, followed by ONE binary frame with the PCM. A phrase the
#: hub could not synthesize comes back with ``error`` and no audio, so the
#: client never plays silence as if it were a cached line.
MSG_TTS_PHRASE = "tts_phrase"
#: v1.7: who the server just recognised by voice, ``{"name": str, "score":
#: float}``. Sent right after identification, before the reply is produced, so
#: the HUD can show the name while the person is still looking at it. An empty
#: name clears it; a voice that matched nobody is simply not announced.
MSG_SPEAKER = "speaker"
MSG_ERROR = "error"
#: v2: the room's settings changed on the hub (``config_update``, ТЗ 4.7).
MSG_CONFIG_UPDATE = "config_update"

#: Hub -> client frames of the background class (ТЗ 13): when a client cannot
#: keep up with the send queue, these are dropped first. A reply's text and its
#: PCM chunks are never in this set.
BACKGROUND_SERVER_MESSAGE_TYPES = frozenset(
    {"camera_state", "camera_frame", "device_state", "hud", "status", "speaker",
     "hub_status", "offline_hint", "card"}
)


def is_background_server_frame(payload: Mapping[str, Any]) -> bool:
    """True when a hub -> client frame may be dropped under backpressure."""
    return str(payload.get("type", "")) in BACKGROUND_SERVER_MESSAGE_TYPES

#: v1.7: optional ``say`` field - a caption the client shows on the HUD for the
#: follow-up window that reply opens (voice enrollment progress), so the person
#: can SEE what Rowan is waiting for instead of guessing from the speech alone.
SAY_STATUS_FIELD = "status"
#: How long a status caption stays up when the server gives no ttl.
DEFAULT_STATUS_TTL_S = 8.0
#: How long a card stays on the HUD when the server gives no ttl (F-709: cards
#: auto-hide; a card is a moment, not a window).
DEFAULT_CARD_TTL_S = 30.0

# --- shared literals used inside the frames ---------------------------------
#: WebSocket endpoint path served by the brain server.
WS_PATH = "/ws"

#: Audio wire format for both microphone and TTS streams.
AUDIO_FORMAT = "pcm_s16le"

#: Channel count for both directions (mono).
AUDIO_CHANNELS = 1

#: Microphone sample rate expected by the server (Whisper/VAD/Vosk all use it).
MIC_SAMPLE_RATE = 16000

#: Image format of the screenshot frame (SPEC section 4, client message 6).
SCREENSHOT_FORMAT = "jpeg"

#: Image format of a camera frame (v1.4) -- the same JPEG wire format.
CAMERA_FORMAT = "jpeg"

#: v1.4: a camera frame the client pushed on its own while somebody is visible.
CAMERA_REASON_PRESENCE = "presence"

#: v1.4: a camera frame answering a :data:`MSG_CAMERA_REQUEST`.
CAMERA_REASON_REQUEST = "request"

#: Burst extension (v1.4): default/absent ``burst`` on a ``camera_request`` -
#: a single frame, ``seq=1, of=1`` -- unchanged behaviour for old call sites.
CAMERA_BURST_DEFAULT = 1

#: Burst extension (v1.4): the most frames one ``camera_request`` may ask for
#: (and the most a presence mini-burst push carries), so a misbehaving caller
#: cannot turn one request into an unbounded stream of frames.
CAMERA_BURST_MAX = 5

# Optional hello capability. A clip is one MP4 binary frame, never PCM audio.
CAP_CAMERA_CLIP = "camera_clip"
#: The room can show its own camera as live video on its own screen.
CAP_CAMERA_PREVIEW = "camera_preview"
CAMERA_CLIP_MAX_BYTES = 20_000_000

#: Error message the server sends when STT produced nothing (false wake-word).
ERR_EMPTY_TRANSCRIPT = "empty transcript"

#: Tool result the server substitutes when the client does not answer in time.
ERR_CLIENT_TIMEOUT = "client timeout"

#: Every message type a client may send.
CLIENT_MESSAGE_TYPES = frozenset(
    {
        MSG_HELLO,
        MSG_ROOM_SPEECH,
        MSG_UTTERANCE_START,
        MSG_UTTERANCE_END,
        MSG_UTTERANCE_TEXT,
        MSG_ACTION_RESULT,
        MSG_VOICE_CONFIRMATION_RESULT,
        MSG_SCREENSHOT,
        MSG_SCREENSHOT_ERROR,
        MSG_CAMERA_STATE,
        MSG_TRACKS,
        MSG_BODY_CROP,
        MSG_CAMERA_FRAME,
        MSG_CAMERA_ERROR,
        MSG_ROOM_HEALTH,
        MSG_CAMERA_CLIP,
        MSG_CAMERA_CLIP_ERROR,
        MSG_OBJECT_EVENT,
        MSG_POINT_EVENT,
        MSG_POSTURE_EVENT,
        MSG_COMPUTER_USE_STEP,
        MSG_TTS_PREFETCH,
    }
)

#: Every message type a server may send.
MSG_CHAT = "chat"
MSG_INTERRUPT_REQUEST = "interrupt_request"
MSG_NOTICE = "notice"
MSG_DISMISS = "dismiss"
MSG_DISMISSED = "dismissed"
CLIENT_MESSAGE_TYPES = CLIENT_MESSAGE_TYPES | {MSG_INTERRUPT_REQUEST, MSG_DISMISS}

SERVER_MESSAGE_TYPES = frozenset(
    {
        MSG_READY,
        MSG_CHAT,
        MSG_NOTICE,
        MSG_DISMISSED,
        MSG_TRANSCRIPT,
        MSG_TRANSCRIPT_PARTIAL,
        MSG_VOICE_CONFIRMATION,
        MSG_ACTIONS,
        MSG_SCREENSHOT_REQUEST,
        MSG_CAMERA_REQUEST,
        MSG_CAMERA_CLIP_REQUEST,
        MSG_PRIVACY,
        MSG_SAY,
        MSG_TTS_START,
        MSG_TTS_END,
        MSG_IMAGE_SHOW,
        MSG_PLAY_AUDIO,
        MSG_STATUS,
        MSG_CARD,
        MSG_HUB_STATUS,
        MSG_SPEAKER,
        MSG_OFFLINE_HINT,
        MSG_TTS_PHRASE,
        MSG_ERROR,
        MSG_CONFIG_UPDATE,
        MSG_COMPUTER_USE,
    }
)

#: ТЗ F-711: a phone client has no camera and no PC, so these frames cannot
#: come from it. The hub refuses them instead of pretending to see a room.
PHONE_FORBIDDEN_INPUTS = frozenset(
    {
        MSG_CAMERA_STATE,
        MSG_TRACKS,
        MSG_BODY_CROP,
        MSG_CAMERA_FRAME,
        MSG_CAMERA_ERROR,
        MSG_CAMERA_CLIP,
        MSG_CAMERA_CLIP_ERROR,
        MSG_OBJECT_EVENT,
        MSG_POINT_EVENT,
        MSG_POSTURE_EVENT,
        MSG_COMPUTER_USE_STEP,
        MSG_SCREENSHOT,
        MSG_SCREENSHOT_ERROR,
    }
)

#: ТЗ F-711: client tools a phone cannot run - they all act on a computer.
PHONE_FORBIDDEN_TOOLS = frozenset({"pc_control", "run_command", "browser_control"})

__all__ = [
    "MSG_DISMISS",
    "MSG_DISMISSED",
    "MSG_CHAT",
    "MSG_NOTICE",
    "MSG_INTERRUPT_REQUEST",
    "MSG_HELLO",
    "MSG_ROOM_SPEECH",
    "MSG_UTTERANCE_START",
    "MSG_UTTERANCE_END",
    "MSG_UTTERANCE_TEXT",
    "MSG_ACTION_RESULT",
    "MSG_SCREENSHOT",
    "MSG_SCREENSHOT_ERROR",
    "MSG_CAMERA_STATE",
    "MSG_TRACKS",
    "MSG_BODY_CROP",
    "MSG_CAMERA_FRAME",
    "MSG_CAMERA_ERROR",
    "MSG_ROOM_HEALTH",
    "MSG_CAMERA_CLIP",
    "MSG_CAMERA_CLIP_ERROR",
    "MSG_CAMERA_CLIP_REQUEST",
    "MSG_SOUND_EVENT",
    "MSG_OBJECT_EVENT",
    "MSG_POINT_EVENT",
    "MSG_POSTURE_EVENT",
    "MSG_COMPUTER_USE",
    "MSG_COMPUTER_USE_STEP",
    "MSG_PRIVACY",
    "MSG_OFFLINE_HINT",
    "MSG_TTS_PREFETCH",
    "MSG_TTS_PHRASE",
    "CAP_CAMERA_CLIP",
    "CAP_CAMERA_PREVIEW",
    "MSG_CAMERA_PREVIEW",
    "CAMERA_CLIP_MAX_BYTES",
    "MSG_READY",
    "MSG_TRANSCRIPT",
    "MSG_ACTIONS",
    "MSG_SCREENSHOT_REQUEST",
    "MSG_CAMERA_REQUEST",
    "MSG_SAY",
    "MSG_TTS_START",
    "MSG_TTS_END",
    "MSG_IMAGE_SHOW",
    "MSG_PLAY_AUDIO",
    "MSG_STATUS",
    "MSG_CARD",
    "DEFAULT_CARD_TTL_S",
    "MSG_HUB_STATUS",
    "MSG_SPEAKER",
    "MSG_CONFIG_UPDATE",
    "BACKGROUND_SERVER_MESSAGE_TYPES",
    "is_background_server_frame",
    "SAY_STATUS_FIELD",
    "DEFAULT_STATUS_TTL_S",
    "MSG_ERROR",
    "WS_PATH",
    "AUDIO_FORMAT",
    "AUDIO_CHANNELS",
    "MIC_SAMPLE_RATE",
    "SCREENSHOT_FORMAT",
    "CAMERA_FORMAT",
    "CAMERA_REASON_PRESENCE",
    "CAMERA_REASON_REQUEST",
    "CAMERA_BURST_DEFAULT",
    "CAMERA_BURST_MAX",
    "ERR_EMPTY_TRANSCRIPT",
    "ERR_CLIENT_TIMEOUT",
    "CLIENT_MESSAGE_TYPES",
    "SERVER_MESSAGE_TYPES",
    "PHONE_FORBIDDEN_INPUTS",
    "PHONE_FORBIDDEN_TOOLS",
]


# ---------------------------------------------------------------------------
# Protocol v2 (ТЗ section 13)
# ---------------------------------------------------------------------------
# Every v2 text frame carries the same envelope: type, proto (=2), ts (ms),
# home_id, client_id, seq, plus an optional utterance_id / event_id. The models
# are strict (extra="forbid") and are collected into a discriminated union on
# `type`, so an unknown frame is reported as a protocol error instead of being
# guessed at. Binary frames keep the v1 rule: a binary payload always follows
# the header that announced it, in the same session, with nothing in between.

#: Current protocol version written into ``hello_ok``.
PROTOCOL_VERSION = 2
#: Frames without a ``proto`` field are v1 and stay accepted until phase 2 ends.
LEGACY_PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """A frame could not be parsed; the caller answers with an ``error`` frame."""


class ClientKind(str, Enum):
    ROOM_PC = "room_pc"
    PHONE = "phone"
    SENSOR_NODE = "sensor_node"


class Role(str, Enum):
    ADMIN = "admin"
    TRUSTED = "trusted"
    USER = "user"
    GUEST = "guest"


def _now_ms() -> int:
    return int(time.time() * 1000)


class Envelope(BaseModel):
    """Fields every v2 text frame carries (ТЗ section 13)."""

    model_config = ConfigDict(extra="forbid")

    proto: Literal[2] = 2
    ts: int = Field(default_factory=_now_ms, ge=0)
    home_id: str = Field(default="", max_length=100)
    client_id: str = Field(default="", max_length=100)
    seq: int = Field(default=0, ge=0)
    utterance_id: str | None = Field(default=None, max_length=100)
    event_id: str | None = Field(default=None, max_length=100)


class Track(BaseModel):
    """One person track reported by a client (F-201)."""

    model_config = ConfigDict(extra="forbid")

    track_id: str = Field(min_length=1, max_length=100)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    conf: float = Field(default=0.0, ge=0.0, le=1.0)
    zone: str = Field(default="", max_length=100)
    since: float = 0.0


class ActionItem(BaseModel):
    """One tool call the client must execute."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    kind: str = Field(min_length=1, max_length=100)
    args: dict[str, Any] = Field(default_factory=dict)


# --- client -> server -------------------------------------------------------


class Hello(Envelope):
    type: Literal["hello"] = "hello"
    token: str = Field(default="", max_length=200)
    kind: ClientKind = ClientKind.ROOM_PC
    caps: list[str] = Field(default_factory=list)
    version: str = Field(default="", max_length=50)
    hw: str = Field(default="", max_length=200)
    #: ТЗ F-303: privacy mode lives on the CLIENT, so a client that restarted
    #: says what it really has and the hub believes it over its own memory.
    privacy: bool = False


class UtteranceStart(Envelope):
    type: Literal["utterance_start"] = "utterance_start"
    sample_rate: int = Field(default=MIC_SAMPLE_RATE, gt=0)
    channels: int = Field(default=AUDIO_CHANNELS, gt=0)
    pre_roll_ms: int = Field(default=0, ge=0)
    #: F-103: ``True`` when this utterance arrived inside the client's own
    #: follow-up window (no wake word was said), ``False`` when it did not,
    #: ``None`` for a client that predates the window and must keep its old
    #: behaviour. The hub asks D-02/D-11 about the turn either way.
    followup: bool | None = None


class UtteranceEnd(Envelope):
    type: Literal["utterance_end"] = "utterance_end"


class UtteranceText(Envelope):
    """ТЗ F-711: the finished words of a client that did its own VAD and STT."""

    type: Literal["utterance_text"] = "utterance_text"
    text: str = Field(min_length=1, max_length=2000)
    #: What the client's own recognizer heard, if it knows (a whisper code).
    language: str = Field(default="", max_length=16)


class ActionResult(Envelope):
    type: Literal["action_result"] = "action_result"
    action_id: str = Field(min_length=1, max_length=100)
    ok: bool
    detail: str = Field(default="", max_length=2000)


class Tracks(Envelope):
    type: Literal["tracks"] = "tracks"
    tracks: list[Track] = Field(default_factory=list)


class BodyCropHeader(Envelope):
    type: Literal["body_crop"] = "body_crop"
    track_id: str = Field(min_length=1, max_length=100)
    kind: Literal["body", "face"] = "body"
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)


class FaceBurstHeader(Envelope):
    type: Literal["face_burst"] = "face_burst"
    track_id: str = Field(default="", max_length=100)
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)


class ScreenshotHeader(Envelope):
    type: Literal["screenshot"] = "screenshot"
    format: Literal["jpeg"] = "jpeg"


class CameraFrameHeader(Envelope):
    type: Literal["camera_frame"] = "camera_frame"
    reason: Literal["presence", "request"] = "request"
    index: int = Field(default=1, ge=1)
    of: int = Field(default=1, ge=1)
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)


class CameraClipHeader(Envelope):
    type: Literal["camera_clip"] = "camera_clip"
    format: Literal["mp4"] = "mp4"
    bytes: int = Field(default=0, ge=0)
    w: int = Field(default=0, ge=0)
    h: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0.0)
    fps: int = Field(default=0, ge=0)


class SoundEvent(Envelope):
    type: Literal["sound_event"] = "sound_event"
    label: str = Field(min_length=1, max_length=100)
    conf: float = Field(default=0.0, ge=0.0, le=1.0)
    at_ms: int = Field(default=0, ge=0)


class DeviceState(Envelope):
    type: Literal["device_state"] = "device_state"
    device_id: str = Field(min_length=1, max_length=100)
    capability: str = Field(min_length=1, max_length=100)
    value: Any = None


class BargeIn(Envelope):
    type: Literal["barge_in"] = "barge_in"
    say_id: str = Field(default="", max_length=100)
    at_ms: int = Field(default=0, ge=0)


class Ping(Envelope):
    type: Literal["ping"] = "ping"


class Pong(Envelope):
    type: Literal["pong"] = "pong"


# --- server -> client -------------------------------------------------------


class HelloOk(Envelope):
    type: Literal["hello_ok"] = "hello_ok"
    session_id: str = Field(min_length=1, max_length=100)
    server_version: str = Field(default="", max_length=50)
    config_rev: int = Field(default=1, ge=0)


class HelloErr(Envelope):
    type: Literal["hello_err"] = "hello_err"
    code: int = 4401
    message: str = Field(default="", max_length=500)


class Transcript(Envelope):
    type: Literal["transcript"] = "transcript"
    text: str = ""
    partial: bool = False
    language: str = Field(default="", max_length=20)
    speaker: str = Field(default="", max_length=100)


class Actions(Envelope):
    type: Literal["actions"] = "actions"
    actions: list[ActionItem] = Field(default_factory=list)


class Say(Envelope):
    type: Literal["say"] = "say"
    say_id: str = Field(default="", max_length=100)
    text: str = ""
    voice: str = Field(default="", max_length=100)
    volume: float | None = Field(default=None, ge=0.0, le=1.0)
    interruptible: bool = True


class TtsStart(Envelope):
    type: Literal["tts_start"] = "tts_start"
    say_id: str = Field(default="", max_length=100)
    sample_rate: int = Field(default=48000, gt=0)
    format: Literal["pcm_s16le"] = "pcm_s16le"
    channels: int = Field(default=1, gt=0)


class TtsEnd(Envelope):
    type: Literal["tts_end"] = "tts_end"
    say_id: str = Field(default="", max_length=100)


class ListenFollowup(Envelope):
    type: Literal["listen_followup"] = "listen_followup"
    window_ms: int = Field(default=0, ge=0)


class Identity(Envelope):
    type: Literal["identity"] = "identity"
    track_id: str = Field(min_length=1, max_length=100)
    person_id: str | None = Field(default=None, max_length=100)
    name: str | None = Field(default=None, max_length=120)
    p: float = Field(default=0.0, ge=0.0, le=1.0)
    role: Role = Role.USER


class CameraRequest(Envelope):
    type: Literal["camera_request"] = "camera_request"
    kind: Literal["frame", "clip"] = "frame"
    zone: str = Field(default="", max_length=100)
    burst: int = Field(default=CAMERA_BURST_DEFAULT, ge=1, le=CAMERA_BURST_MAX)
    full: bool = False
    clip_seconds: int = Field(default=0, ge=0, le=60)


class ScreenshotRequest(Envelope):
    type: Literal["screenshot_request"] = "screenshot_request"


class CameraClipRequest(Envelope):
    type: Literal["camera_clip_request"] = "camera_clip_request"
    # ТЗ F-702: one video of an alert episode may be up to a minute long.
    seconds: int = Field(default=5, ge=3, le=60)
    fps: int = Field(default=8, ge=5, le=10)


class Privacy(Envelope):
    """Server → client: the camera of this room goes off, or comes back (F-303)."""

    type: Literal["privacy"] = "privacy"
    on: bool = True
    reason: str = Field(default="", max_length=100)


class DeviceSet(Envelope):
    type: Literal["device_set"] = "device_set"
    device_id: str = Field(min_length=1, max_length=100)
    capability: str = Field(min_length=1, max_length=100)
    value: Any = None
    action_id: str = Field(default="", max_length=100)


class Hud(Envelope):
    type: Literal["hud"] = "hud"
    kind: Literal["state", "card", "overlay"] = "state"
    payload: dict[str, Any] = Field(default_factory=dict)


class ConfigUpdate(Envelope):
    type: Literal["config_update"] = "config_update"
    config_rev: int = Field(default=1, ge=0)
    patch: dict[str, Any] = Field(default_factory=dict)


class OfflineHint(Envelope):
    type: Literal["offline_hint"] = "offline_hint"
    reason: str = Field(default="", max_length=200)
    eta_s: float = Field(default=0.0, ge=0.0)


class Intercom(Envelope):
    type: Literal["intercom"] = "intercom"
    from_person: str = Field(default="", max_length=120)
    text: str = ""
    audio: str | None = Field(default=None, max_length=500)


class ErrorMessage(Envelope):
    type: Literal["error"] = "error"
    code: int = 0
    message: str = Field(default="", max_length=500)
    ref_seq: int | None = Field(default=None, ge=0)


ClientMessage = Annotated[
    Hello | UtteranceStart | UtteranceEnd | UtteranceText | ActionResult | Tracks | BodyCropHeader | FaceBurstHeader | ScreenshotHeader | CameraFrameHeader | CameraClipHeader | SoundEvent | DeviceState | BargeIn | Ping | Pong,
    Field(discriminator="type"),
]
ServerMessage = Annotated[
    HelloOk | HelloErr | Transcript | Actions | Say | TtsStart | TtsEnd | ListenFollowup | Identity | CameraRequest | ScreenshotRequest | CameraClipRequest | DeviceSet | Hud | ConfigUpdate | OfflineHint | Intercom | ErrorMessage,
    Field(discriminator="type"),
]

_CLIENT_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
_SERVER_ADAPTER: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def protocol_version(raw: Mapping[str, Any]) -> int:
    """The frame's protocol version; frames without ``proto`` are v1."""
    value = raw.get("proto", LEGACY_PROTOCOL_VERSION)
    if isinstance(value, bool) or not isinstance(value, int):
        return LEGACY_PROTOCOL_VERSION
    return value


def parse_message(raw: Mapping[str, Any], *, direction: str = "client") -> BaseModel | None:
    """Parse a v2 frame; return ``None`` for a v1 frame (caller keeps v1 path).

    Raises :class:`ProtocolError` for an unknown or malformed v2 frame. The
    caller reports it as an ``error`` frame and keeps the connection open.
    """
    if not isinstance(raw, Mapping):
        raise ProtocolError("A control frame must be a JSON object.")
    if protocol_version(raw) < PROTOCOL_VERSION:
        return None
    if direction not in {"client", "server"}:
        raise ProtocolError(f"Unknown direction {direction!r}.")
    adapter = _CLIENT_ADAPTER if direction == "client" else _SERVER_ADAPTER
    try:
        return adapter.validate_python(dict(raw))
    except ValidationError as exc:
        raise ProtocolError(str(exc)) from None
