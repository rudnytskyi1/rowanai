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

# --- client -> server -------------------------------------------------------
MSG_HELLO = "hello"
#: Idle local VAD heartbeat, no audio/transcript: defer unsolicited greetings.
MSG_ROOM_SPEECH = "room_speech"
MSG_UTTERANCE_START = "utterance_start"
MSG_UTTERANCE_END = "utterance_end"
MSG_ACTION_RESULT = "action_result"
#: Reply to a physical confirmation on the room PC; never an LLM tool result.
MSG_VOICE_CONFIRMATION_RESULT = "voice_confirmation_result"
#: v1.1: header announcing the single binary frame with the JPEG screenshot.
MSG_SCREENSHOT = "screenshot"
#: v1.1: the client could not capture the screen; no binary frame follows.
MSG_SCREENSHOT_ERROR = "screenshot_error"
#: v1.4: what the room camera currently sees (people count + object counts).
MSG_CAMERA_STATE = "camera_state"
#: v1.4: header announcing the single binary frame with a camera JPEG.
MSG_CAMERA_FRAME = "camera_frame"
#: v1.4: the client could not grab a camera frame; no binary frame follows.
MSG_CAMERA_ERROR = "camera_error"
MSG_CAMERA_CLIP = "camera_clip"
MSG_CAMERA_CLIP_ERROR = "camera_clip_error"

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
MSG_SAY = "say"
MSG_TTS_START = "tts_start"
MSG_TTS_END = "tts_end"
#: v1.6: show a photo on the room screen (header, then one binary JPEG frame).
MSG_IMAGE_SHOW = "image_show"
#: v1.7: a short caption for the room screen, ``{"text": str, "ttl_s": float}``,
#: shown on the HUD while something slow happens in the background (face
#: enrollment photos). Empty text clears it. Nothing is spoken.
MSG_STATUS = "status"
#: v1.7: who the server just recognised by voice, ``{"name": str, "score":
#: float}``. Sent right after identification, before the reply is produced, so
#: the HUD can show the name while the person is still looking at it. An empty
#: name clears it; a voice that matched nobody is simply not announced.
MSG_SPEAKER = "speaker"
MSG_ERROR = "error"

#: v1.7: optional ``say`` field - a caption the client shows on the HUD for the
#: follow-up window that reply opens (voice enrollment progress), so the person
#: can SEE what Rowan is waiting for instead of guessing from the speech alone.
SAY_STATUS_FIELD = "status"
#: How long a status caption stays up when the server gives no ttl.
DEFAULT_STATUS_TTL_S = 8.0

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
        MSG_ACTION_RESULT,
        MSG_VOICE_CONFIRMATION_RESULT,
        MSG_SCREENSHOT,
        MSG_SCREENSHOT_ERROR,
        MSG_CAMERA_STATE,
        MSG_CAMERA_FRAME,
        MSG_CAMERA_ERROR,
        MSG_CAMERA_CLIP,
        MSG_CAMERA_CLIP_ERROR,
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
        MSG_SAY,
        MSG_TTS_START,
        MSG_TTS_END,
        MSG_IMAGE_SHOW,
        MSG_STATUS,
        MSG_SPEAKER,
        MSG_ERROR,
    }
)

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
    "MSG_ACTION_RESULT",
    "MSG_SCREENSHOT",
    "MSG_SCREENSHOT_ERROR",
    "MSG_CAMERA_STATE",
    "MSG_CAMERA_FRAME",
    "MSG_CAMERA_ERROR",
    "MSG_CAMERA_CLIP",
    "MSG_CAMERA_CLIP_ERROR",
    "MSG_CAMERA_CLIP_REQUEST",
    "CAP_CAMERA_CLIP",
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
    "MSG_STATUS",
    "MSG_SPEAKER",
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
]
